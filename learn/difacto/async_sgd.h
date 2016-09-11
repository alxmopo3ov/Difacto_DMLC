#pragma once
#include "progress.h"
#include "config.pb.h"
#include "loss.h"
#include "base/localizer.h"
#include "solver/minibatch_solver.h"
#include <array>
#include <string>
#include "dmlc/logging.h"

namespace dmlc {
namespace difacto {



#define BIAS_KEY 14437434782623107211ull


/**
* \brief the scheduler for async SGD
*/
class AsyncScheduler : public solver::MinibatchScheduler {
public:
    AsyncScheduler(const Config& conf) : conf_(conf) {
        if (conf_.early_stop()) {
            CHECK(conf_.val_data().size()) << "early stop needs validation dataset";
        }
        Init(conf);
    }
    virtual ~AsyncScheduler() { }

    virtual std::string ProgHeader() { return Progress::HeadStr(); }

    virtual std::string ProgString(const solver::Progress& prog) {
        if (!prog.empty()) {
            prog_.data = prog;
        }
        return prog_.PrintStr();
    }

    virtual bool Stop(const solver::Progress& cur_, bool train) {
        difacto::Progress cur;
        cur.data = cur_;
        double cur_objv = cur.objv() / cur.new_ex();
        if (train) {
            if (conf_.has_max_objv() && cur_objv > conf_.max_objv()) {
                return true;
            }
        } else {
            double diff = pre_val_objv_ - cur_objv;
            pre_val_objv_ = cur_objv;
            if (conf_.early_stop() && diff < conf_.min_objv_decr()) {
                std::cout << "The decrease of validation objective "
                << "is smaller than the minimal requirement: "
                << diff << " vs " << conf_.min_objv_decr()
                << std::endl;
                return true;
            }
        }
        return false;
    }

private:
    Progress prog_;
    Config conf_;
    double pre_val_objv_ = 100;
};

using FeaID = ps::Key;
template <typename T> using Blob = ps::Blob<T>;
static const int kPushFeaCnt = 1;

/**
* \brief the base sgd handle
*/
struct ISGDHandle {
    ISGDHandle() { ns_ = ps::NodeInfo::NumServers(); }
    inline void Start(bool push, int timestamp, int cmd, void* msg) {
        push_count = push && (cmd == kPushFeaCnt) ? true : false;
        perf_.Start(push, cmd);
    }


    inline void Report() {
        // reduce communication frequency
        ++ ct_;
        if (ct_ >= ns_ && reporter) {
            Progress prog; prog.new_w() = new_w; prog.new_V() = new_V; reporter(prog);
            new_w = 0; new_V = 0;ct_ = 0;
        }
    }

    inline void Finish() { Report(); perf_.Stop(); }

    // for w
    float lambda_l1 = 0, lambda_l2 = 0;
    float alpha = .01, beta = 1;
    Config::Algo_W algo_w = Config::FTRL_W;

    // for V
    struct Embedding {
        Config::Embedding::Algo_V algo_v = Config::Embedding::ADAGRAD_V;
        int dim = 0;
        unsigned thr, thr_step;
        float lambda_l1 = 0, lambda_l1_incremental = 0;
        float lambda_l2 = 0, lambda_l2_incremental = 0;
        float alpha = .01, beta = 1;
        float V_min = -0.01f, V_max = .01;
        float lambda_l1_2 = 0, lambda_l1_2_incremental = 0;
        float lr_nu = 0.999;
        float momentum_mu = 0.9;
        bool l1_2_only_small = true;
    };
    Embedding V;
    bool l1_shrk;
    bool learn_bias_embedding;

    // statistic
    bool push_count;
    static int64_t new_w;
    static int64_t new_V;
    std::function<void(const Progress& prog)> reporter;

    void Load(Stream* fi) { }
    void Save(Stream *fo) const { }

    bool NeedAdd(bool is_new) {
        return true;
    }
private:
    // performance monitor and logger
    class Perf {
    public:
        void Start(bool push, int cmd) {
            time_[0] = GetTime();
            i_ = push ? ((cmd == kPushFeaCnt) ? 1 : 2) : 3;
        }
        void Stop() {
            time_[i_] += GetTime() - time_[0];
            ++ count_[i_]; ++ count_[0];
            if ((count_[0] % disp_) == 0) {
                LOG(INFO) << "push feacnt: " << count_[1] << " x " << time_[1]/count_[1]
                << ", push grad: " << count_[2] << " x " << time_[2]/count_[2]
                << ", pull: " << count_[3] << " x " << time_[3]/count_[3];
            }
        }
    private:
        std::array<double, 4> time_{};
        std::array<int, 4> count_{};
        int i_ = 0, disp_ = ps::NodeInfo::NumWorkers() * 10;
    } perf_;

    int ct_ = 0, ns_ = 0;
};

/**
* \brief value stored on server nodes
*/
struct AdaGradEntry {
    AdaGradEntry() { }
    ~AdaGradEntry() { Clear(); }

    inline void Clear() {
        if ( size > 1 ) {
            delete [] w;
            delete [] sqc_grad;
            delete [] z_V;
            delete [] nag_prev;
        }
        size = 0;
        w = NULL;
        sqc_grad = NULL;
        z_V = NULL;
        nag_prev = NULL;
    }

    inline void Resize(int n) {
        if (n < size) {
            size = n;
            return;
        }

        float* new_w = new float[n];
        float* new_cg = new float[n+1];
        float* new_z_V = new float[n-1];
        float* new_nag_prev = new float[n-1];

        if (size == 1) {
            new_w[0] = w_0();
            new_cg[0] = sqc_grad_0();
            new_cg[1] = z_0();
        } else {
            memcpy(new_w, w, size * sizeof(float));
            memcpy(new_cg, sqc_grad, (size+1) * sizeof(float));
            memcpy(new_z_V, z_V, (size-1) * sizeof(float));
            memcpy(new_nag_prev, nag_prev, (size-1) * sizeof(float));
            Clear();
        }
        w = new_w;
        sqc_grad = new_cg;
        z_V = new_z_V;
        nag_prev = new_nag_prev;
        size = n;
    }

    inline float& w_0() { return size == 1 ? *(float *)&w : w[0]; }
    inline float w_0() const { return size == 1 ? *(float *)&w : w[0]; }

    // Pure L1/2 regularization is intractable, because in this case we must
    // initialize all weight first and store them in memory. Hence, there is no
    // good alternative for l1-shrinking as for memory constraints heuristic.
    //
    // However, we can still use L1/2 as additional regularization term for
    // already initialized embeddings.

    inline float& sqc_grad_0() {
        return size == 1 ? *(float *)&sqc_grad : sqc_grad[0];
    }

    inline float& z_0() {
        return size == 1 ? *(((float *)&sqc_grad)+1) : sqc_grad[1];
    }

    void Load(Stream* fi, bool full_state_mode = false) {
        fi->Read(&size, sizeof(size)) ;
        if (size == 1) {
            fi->Read((float *)&w, sizeof(float));
            if(full_state_mode) {
                fi->Read(&sqc_grad, sizeof(float*));
            }
        } else {
            w = new float[size];
            sqc_grad = new float[size+1];
            fi->Read(w, sizeof(float)*size);
            if(full_state_mode) {
                fi->Read(sqc_grad, sizeof(float)*(size+1));
            } else {
                memset(sqc_grad, 0, sizeof(float) * (size + 1));
            }
            ISGDHandle::new_V += size - 1;
        }
        if (w_0() != 0) ++ ISGDHandle::new_w;
    }

    void Save(Stream *fo, bool full_state_mode = false) const {
        fo->Write(&size, sizeof(size));
        if (size == 1) {
            fo->Write((float *)&w, sizeof(float));
            if(full_state_mode) {
                fo->Write(&sqc_grad, sizeof(float*));
            }
        } else {
            fo->Write(w, sizeof(float)*size);
            if(full_state_mode) {
                fo->Write(sqc_grad, sizeof(float)*(size+1));
            }
        }
    }

    bool Empty() const { return (w_0() == 0 && size == 1); }

    /// #appearence of this feature in the data
    unsigned fea_cnt = 0;

    /// FIXME: Experimental. Number of occurences of this feature in minibatch
    unsigned minibatch_occurence_count = 0;

    /// length of w. if size == 1, then using w itself to store the value to save
    /// memory and avoid unnecessary new (see w_0())
    int size = 1;

    /// w and V
    float *w = NULL;

    /// z values fow V.
    /// FIXME: this is redundant and must be replaced with more memory-efficient storing only z
    float *z_V = NULL;

    /// This is storage for prevous weights for nesterov accelerated gradient with reverse proximal function
    /// FIXME: Too much memory required!
    float *nag_prev = NULL;

    /// square root of the cumulative gradient
    float *sqc_grad = NULL;

    friend bool operator<(const AdaGradEntry & a, const AdaGradEntry & b) {
        return fabs(a.w_0()) < fabs(b.w_0());
    }

    // TEMPORARY!
    float momentum_mu_power = 1.0f;
    float lr_nu_power = 1.0f;

    // THIS IS DUPLICATE OF PREVOUS PARAMETERS!! Because w and V meet simultaneously
    float lr_nu_power_w = 1.0f;
    float momentum_mu_power_w = 1.0f;

    bool is_active_embedding = false;
};

/**
* \brief model updater
*/
struct AdaGradHandle : public ISGDHandle {

    inline bool Push(FeaID key, Blob<const float> recv, AdaGradEntry& val, bool is_new) {
        if (push_count) {
            // FIXME! Interesting solution... recv[0] is gradient. In case of 0-1 it is true, but
            // if x != 0-1 it will fail. Remember it
            // Maybe (unsigned) (recv[0] != 0) will be more proper
            val.fea_cnt += (unsigned) recv[0];
            Resize(val, key);
        } else {
            val.minibatch_occurence_count++;
            CHECK_GE(recv.size, (size_t)0);

            UpdateW(val, recv[0], key);

            // Consistency between sizes
            if (recv.size > 1 && val.size > 1) {
                CHECK_LE(recv.size, (size_t)val.size);
                UpdateV(val, recv);
            }
        }
        return NeedAdd(is_new);
    }

    inline void Pull(FeaID key, const AdaGradEntry& val, Blob<float>& send) {
        float w0 = val.w_0();
        if (val.size == 1 || (l1_shrk && (w0 == 0))) {
            CHECK_GT(send.size, (size_t)0);
            send[0] = w0;
            send.size = 1;
        } else {
            send.data = val.w;
            send.size = val.size;
        }
    }

    // FIXME: Dirty hack to force DiFacto not to learn embedding for bias
    /// \brief resize if necessary
    inline void Resize(AdaGradEntry& val, FeaID key) {
        // resize the larger dim first to avoid double resize
        if (val.fea_cnt >= V.thr && val.size < V.dim + 1 &&
            (!l1_shrk || val.w_0() != 0) && (learn_bias_embedding || key != (FeaID)BIAS_KEY)) {

            int old_siz = val.size;
            if (!V.thr_step) {
                val.Resize(V.dim + 1);
            } else {
                val.Resize(std::min(val.fea_cnt + 1, std::min(val.size + V.thr_step, (unsigned)V.dim + 1)));
            }

            for (int j = old_siz; j < val.size; ++j) {
                val.w[j] = rand() / (float) RAND_MAX * (V.V_max - V.V_min) + V.V_min;
                val.sqc_grad[j+1] = 0;
            }

            for(int i = old_siz; i < val.size; i++) {
                if (val.z_V) {
                    val.z_V[i - 1] = 0.0;
                }
                if (val.nag_prev) {
                    val.nag_prev[i - 1] = 0.0;
                }
            }

            // We add all embedding if it was not active and add only new weights if it was
            if(val.is_active_embedding) {
                new_V += val.size - old_siz;
            } else {
                new_V += val.size - 1;
                val.is_active_embedding = true;
            }
        }
    }

    inline void recalculate_new_V(unsigned size, bool is_active_now, bool was_active_before)
    {
        if(!is_active_now && was_active_before) {
            new_V -= size;
        } else if(is_active_now && !was_active_before) {
            new_V += size;
        }
    }

    // ftrl
    inline void UpdateW(AdaGradEntry& val, float g, FeaID key) {
        // update w
        if (algo_w == Config::ADAGRAD_W) {
            UpdateW_adagrad(val, g, key);
        } else if (algo_w == Config::FTRL_W) {
            UpdateW_FTRL_My(val, g, key);
        } else if (algo_w == Config::FTRL_dmlc) {
            UpdateW_FTRL_dmlc(val, g, key);
        } else if(algo_w == Config::FTRL_dmlc_RMSProp) {
            UpdateW_FTRL_dmlc_RMSProp(val, g, key);
        }
    }


    // AdaGrad proximal update
    inline void UpdateW_adagrad(AdaGradEntry &val, float g, FeaID key) {
        float w = val.w_0();

        float cg = val.sqc_grad_0();
        val.sqc_grad_0() = (float) sqrt(cg * cg + g * g );
        float eta = alpha / (beta + val.sqc_grad_0());
        val.w_0() = solve_proximal_operator(-g + w / eta, eta, lambda_l1, lambda_l2);

        if(w == 0 && val.w_0() != 0) {
            ++ new_w; Resize(val, key);
        } else if(w != 0 && val.w_0() == 0) {
            -- new_w;
        }
    }

    //FTRL with adaptive addition of regularization
    inline void UpdateW_FTRL_My(AdaGradEntry& val, float g, FeaID key) {
        float w = val.w_0();
        unsigned occ = val.minibatch_occurence_count;

        float cg = val.sqc_grad_0();
        float cg_new = (float)sqrt( cg * cg + g * g );
        val.sqc_grad_0() = cg_new;

        val.z_0() += g - (cg_new - cg) / alpha * w;
        val.w_0() = solve_proximal_operator(- val.z_0(), alpha / (cg_new + beta), lambda_l1 * occ, lambda_l2 * occ);

        if(w == 0 && val.w_0() != 0) {
            ++ new_w; Resize(val, key);
        } else if(w != 0 && val.w_0() == 0) {
            -- new_w;
        }
    }


    inline void UpdateW_FTRL_dmlc(AdaGradEntry& val, float g, FeaID key) {
        float w = val.w_0();

        float cg = val.sqc_grad_0();
        float cg_new = (float)sqrt( cg * cg + g * g );
        val.sqc_grad_0() = cg_new;

        val.z_0() += g - (cg_new - cg) / alpha * w;
        val.w_0() = solve_proximal_operator(- val.z_0(), alpha / (cg_new + beta), lambda_l1, lambda_l2);

        Resize(val, key);
        if(w == 0 && val.w_0() != 0) {
            ++ new_w;
        } else if(w != 0 && val.w_0() == 0) {
            -- new_w;
        }
    }

    inline void UpdateW_FTRL_dmlc_RMSProp(AdaGradEntry& val, float g, FeaID key) {
        val.lr_nu_power_w *= V.lr_nu;
        float w = val.w_0();

        // temporary crutch for initial bias correction term
        float cg = val.sqc_grad_0();
        float n_t_prev;
        if(val.lr_nu_power_w < V.lr_nu) {
            n_t_prev = (float) sqrt((cg / (1.0f - val.lr_nu_power_w / V.lr_nu)));
        } else {
            n_t_prev = 0;
        }
        cg = V.lr_nu * cg + (1.0f - V.lr_nu) * g * g;
        float n_t_cur = (float) sqrt(cg / (1.0f - val.lr_nu_power_w));
        val.sqc_grad_0() = cg;
        val.z_0() += g - (n_t_cur - n_t_prev) / V.alpha * w;
        val.w_0() = solve_proximal_operator(- val.z_0(), alpha / (n_t_cur + beta), lambda_l1, lambda_l2);

        Resize(val, key);
        if(w == 0 && val.w_0() != 0) {
            ++ new_w;
        } else if(w != 0 && val.w_0() == 0) {
            -- new_w;
        }
    }

    // boolean parameter is optional by now
    inline std::vector<float> solve_proximal_operator_group(float *z, float *cg, float l1, float l2, float l1_2,
                                                            size_t n, bool *active)
    {
        double cum_z = .0f;
        for(int i = 0; i < n; i++) {
            cum_z += z[i] * z[i];
        }
        if (sqrt(cum_z) < l1_2 * sqrt(n)) {
            recalculate_new_V(n, false, *active);
            *active = false;
            return std::vector<float> (n);
        } else {
            std::vector<float> w(n);
            float eta = .0f;
            for(int i = 0; i < n; i++) {
                eta = (float) (V.alpha / (cg[i] + V.beta));
                // FIXME! minus!
                w[i] = - (float) ((1.0f / (l2 + 1.0f / eta)) * (1.0f - l1_2 / sqrt(cum_z)) * z[i]);
            }
            recalculate_new_V(n, true, *active);
            *active = true;
            return w;
        }
    }

    // Solves FTRL Proximal Operator and return weight
    inline float solve_proximal_operator(float z, float eta, float l1, float l2)
    {
        float w;
        if (z <= l1  && z >= - l1) {
            w = 0;
        } else {
            w = (z > 0 ? z - l1 : z + l1) / (l2 + 1.0f / eta);
        }
        return w;
    }

    inline void UpdateV(AdaGradEntry& val, Blob<const float> recv)
    {
        if (V.algo_v == Config::Embedding::ADAGRAD_V_LINEARIZED) {
            adagrad_linearized_UpdateV(val.w+1, val.sqc_grad+2, recv.data+1, recv.size-1);
        } else if (V.algo_v == Config::Embedding::ADAGRAD_V) {
            adagrad_proximal_UpdateV(val.w+1, val.sqc_grad+2, recv.data+1, recv.size-1);
        } else if (V.algo_v == Config::Embedding::FTRL) {
            ftrl_UpdateV(val.z_V, val.w+1, val.sqc_grad+2, recv.data+1, recv.size-1,
                         val.minibatch_occurence_count, &val.is_active_embedding);
        } else if (V.algo_v == Config::Embedding::RMSProp) {
            val.lr_nu_power *= V.lr_nu;
            rmsprop_UpdateV(val.w+1, val.sqc_grad+2, recv.data+1, recv.size-1, val.lr_nu_power);
        } else if (V.algo_v == Config::Embedding::ADAM) {
            val.momentum_mu_power *= V.momentum_mu;
            val.lr_nu_power *= V.lr_nu;
            adam_UpdateV(val.z_V, val.w+1, val.sqc_grad+2, recv.data+1, recv.size-1, val.momentum_mu_power, val.lr_nu_power);
        } else if (V.algo_v == Config::Embedding::NAG) {
            val.momentum_mu_power *= V.momentum_mu;
            nag_UpdateV(val.z_V, val.w+1, val.sqc_grad+2, recv.data+1, recv.size-1, val.momentum_mu_power);
        } else if (V.algo_v == Config::Embedding::NAG_prox_momentum) {
            val.momentum_mu_power *= V.momentum_mu;
            nag_reverse_prox_UpdateV(val.nag_prev, val.z_V, val.w+1, val.sqc_grad+2, recv.data+1, recv.size-1, val.momentum_mu_power);
        } else if (V.algo_v == Config::Embedding::MOMENTUM) {
            val.momentum_mu_power *= V.momentum_mu;
            momentum_UpdateV(val.z_V, val.w+1, val.sqc_grad+2, recv.data+1, recv.size-1, val.momentum_mu_power);
        } else if (V.algo_v == Config::Embedding::FTRL_RMSProp) {
            val.lr_nu_power *= V.lr_nu;
            ftrl_rmsprop_UpdateV(val.z_V, val.w+1, val.sqc_grad+2, recv.data+1, recv.size-1,
                                 val.minibatch_occurence_count, val.lr_nu_power,
                                 &val.is_active_embedding);
        } else if (V.algo_v == Config::Embedding::NADAM) {
            val.momentum_mu_power *= V.momentum_mu;
            val.lr_nu_power *= V.lr_nu;
            nadam_UpdateV(val.z_V, val.w+1, val.sqc_grad+2, recv.data+1, recv.size-1, val.momentum_mu_power, val.lr_nu_power);
        } else if (V.algo_v == Config::Embedding::NADAM_prox_momentum) {
            val.momentum_mu_power *= V.momentum_mu;
            val.lr_nu_power *= V.lr_nu;
            nadam_reverse_prox_UpdateV(val.nag_prev, val.z_V, val.w+1, val.sqc_grad+2, recv.data+1, recv.size-1,
                                       val.momentum_mu_power, val.lr_nu_power);
        } else if (V.algo_v == Config::Embedding::FTRL_adam) {
            val.momentum_mu_power *= V.momentum_mu;
            val.lr_nu_power *= V.lr_nu;
            ftrl_adam_UpdateV(val.nag_prev, val.z_V, val.w+1, val.sqc_grad+2, recv.data+1, recv.size-1,
                              val.minibatch_occurence_count, val.momentum_mu_power,
                              val.lr_nu_power, &val.is_active_embedding);
        } else if (V.algo_v == Config::Embedding::FTRL_nadam) {
            val.momentum_mu_power *= V.momentum_mu;
            val.lr_nu_power *= V.lr_nu;
            ftrl_nadam_UpdateV(val.nag_prev, val.z_V, val.w+1, val.sqc_grad+2, recv.data+1, recv.size-1,
                              val.minibatch_occurence_count, val.momentum_mu_power,
                              val.lr_nu_power, &val.is_active_embedding);
        }
    }

    // Classic momentum gradient descence with ADAGRAD learning rates
    inline void momentum_UpdateV(float* m, float* w, float* cg, float const* g, int n, float momentum_mu_power) {
        // bias corrected (requires usual learning rates)
        for (int i = 0; i < n; ++i) {
            float grad = g[i];
            m[i] = V.momentum_mu * m[i] + grad;

            cg[i] = (float) sqrt(cg[i] * cg[i] + grad * grad);
            float eta = V.alpha / (cg[i] + V.beta);
            float bias_correction = (float) ((1.0f - momentum_mu_power) / (1.0f - V.momentum_mu));
            w[i] = solve_proximal_operator(-m[i] / bias_correction + w[i] / eta,
                                           eta, V.lambda_l1, V.lambda_l2);
        }
    }

    // RMSProp stochastic optimization method with bias correction
    inline void rmsprop_UpdateV(float* w, float* cg, float const* g, int n, float lr_nu_power) {
        for (int i = 0; i < n; ++i) {
            float grad = g[i];
            cg[i] = V.lr_nu * cg[i] + (1.0f - V.lr_nu) * grad * grad;
            float n_t = (float) (cg[i] / (1.0f - lr_nu_power));
            float eta = (float) (V.alpha / (sqrt(n_t) + V.beta));
            w[i] = solve_proximal_operator(-g[i] + w[i] / eta, eta, V.lambda_l1, V.lambda_l2);
        }
    }

    // Adam stochastic optimization method
    inline void adam_UpdateV(float* m, float* w, float* cg, float const* g, int n,
                             float momentum_mu_power, float lr_nu_power) {
        for (int i = 0; i < n; ++i) {
            float grad = g[i];
            cg[i] = V.lr_nu * cg[i] + (1.0f - V.lr_nu) * grad * grad;
            float n_t = (float) (cg[i] / (1.0f - lr_nu_power));
            float eta = (float) (V.alpha / (sqrt(n_t) + V.beta));

            m[i] = V.momentum_mu * m[i] + (1.0f - V.momentum_mu) * grad;
            float m_t = (float) (m[i] / (1.0f - momentum_mu_power));
            w[i] = solve_proximal_operator(-m_t + w[i] / eta, eta, V.lambda_l1, V.lambda_l2);
        }
    }

    // Nesterov Accelerated Gradient. Shows superior performance on word2vec
    // it will be good to get bias correction here
    //
    // bias_correction:
    inline void nag_UpdateV(float* m, float* w, float* cg, float const* g, int n, float momentum_mu_power) {
        // bias corrected (and magnitude too)
        for (int i = 0; i < n; ++i) {
            float grad = g[i];
            w[i] += V.alpha / (cg[i] + V.beta) * V.momentum_mu * m[i] / (1.0f - momentum_mu_power) * (1 - V.momentum_mu);
            cg[i] = (float) sqrt(cg[i] * cg[i] + grad * grad);
            float eta = V.alpha / ( cg[i] + V.beta );

            m[i] = V.momentum_mu * m[i] + grad;
            float m_t = m[i] / (1.0f - momentum_mu_power) * (1 - V.momentum_mu);
            w[i] = solve_proximal_operator(-m_t + w[i] / eta, eta, V.lambda_l1, V.lambda_l2);
            w[i] -= eta * V.momentum_mu * m[i] / (1.0f - momentum_mu_power * V.momentum_mu) * (1 - V.momentum_mu);
        }
    }

    inline void nag_reverse_prox_UpdateV(float *prev_w, float* m, float* w, float* cg, float const* g, int n, float momentum_mu_power) {
        // bias corrected (and magnitude too)
        for (int i = 0; i < n; ++i) {
            float grad = g[i];
            w[i] = prev_w[i];
            cg[i] = (float) sqrt(cg[i] * cg[i] + grad * grad);
            float eta = V.alpha / ( cg[i] + V.beta );

            m[i] = V.momentum_mu * m[i] + grad;
            float m_t = m[i] / (1.0f - momentum_mu_power) * (1 - V.momentum_mu);
            w[i] = solve_proximal_operator(-m_t + w[i] / eta, eta, V.lambda_l1, V.lambda_l2);
            prev_w[i] = w[i];
            m_t = V.momentum_mu * m[i] / (1.0f - momentum_mu_power * V.momentum_mu) * (1 - V.momentum_mu);
            w[i] = solve_proximal_operator(-m_t + w[i] / eta, eta, V.lambda_l1, V.lambda_l2);
        }
    }


    //Adam with Nesterov Momentum and bias correction (some experiments must be carried out!)
    // For example, we have to solve reverse proximal operator for this kind of method
    inline void nadam_UpdateV(float* m, float* w, float* cg, float const* g, int n,
                              float momentum_mu_power, float lr_nu_power) {

        // bias corrected (and magnitude too)
        for (int i = 0; i < n; ++i) {
            float grad = g[i];
            w[i] += V.alpha / (cg[i] + V.beta) * V.momentum_mu * m[i] / (1.0f - momentum_mu_power) * (1 - V.momentum_mu);
            // RMSProp learning rate
            cg[i] = V.lr_nu * cg[i] + (1.0f - V.lr_nu) * grad * grad;
            float n_t = (float) (cg[i] / (1.0f - lr_nu_power));
            float eta = (float) (V.alpha / (sqrt(n_t) + V.beta));

            m[i] = V.momentum_mu * m[i] + grad;
            float m_t = m[i] / (1.0f - momentum_mu_power) * (1 - V.momentum_mu);
            w[i] = solve_proximal_operator(-m_t + w[i] / eta, eta, V.lambda_l1, V.lambda_l2);
            w[i] -= eta * V.momentum_mu * m[i] / (1.0f - momentum_mu_power * V.momentum_mu) * (1 - V.momentum_mu);
        }
    }

    inline void nadam_reverse_prox_UpdateV(float *prev_w, float* m, float* w, float* cg, float const* g, int n,
                                           float momentum_mu_power, float lr_nu_power)
    {
        // bias corrected (and magnitude too)
        for (int i = 0; i < n; ++i) {
            float grad = g[i];
            w[i] = prev_w[i];
            // RMSProp learning rate
            cg[i] = V.lr_nu * cg[i] + (1.0f - V.lr_nu) * grad * grad;
            float n_t = (float) (cg[i] / (1.0f - lr_nu_power));
            float eta = (float) (V.alpha / (sqrt(n_t) + V.beta));

            m[i] = V.momentum_mu * m[i] + grad;
            float m_t = m[i] / (1.0f - momentum_mu_power) * (1 - V.momentum_mu);
            w[i] = solve_proximal_operator(-m_t + w[i] / eta, eta, V.lambda_l1, V.lambda_l2);
            prev_w[i] = w[i];
            m_t = V.momentum_mu * m[i] / (1.0f - momentum_mu_power * V.momentum_mu) * (1 - V.momentum_mu);
            w[i] = solve_proximal_operator(-m_t + w[i] / eta, eta, V.lambda_l1, V.lambda_l2);
        }
    }


    // adagrad with approximated L2 regularization This is the original update for factorization machine
    inline void adagrad_linearized_UpdateV(float* w, float* cg, float const* g, int n) {
        for (int i = 0; i < n; ++i) {
            float grad = g[i] + V.lambda_l2 * w[i];
            cg[i] = (float) sqrt(cg[i] * cg[i] + grad * grad);
            float eta = V.alpha / ( cg[i] + V.beta );
            w[i] -= eta * grad;
        }
    }


    // adagrad with proximal L2 regularization. There is no l1 regularization here! Experiments are wrong!!
    inline void adagrad_proximal_UpdateV(float* w, float* cg, float const* g, int n) {
        for (int i = 0; i < n; ++i) {
            float grad = g[i];
            cg[i] = (float) sqrt(cg[i] * cg[i] + grad * grad);
            float eta = V.alpha / ( cg[i] + V.beta );
            w[i] = solve_proximal_operator(-g[i] + w[i] / eta, eta, V.lambda_l1, V.lambda_l2);
        }
    }


    // FTRL Proximal My update
    // FIXME: Currently we store both z and w, but instead we can store only z
    // FIXME: Produces different learning curves in comparision with adagrad update. This is wrong!
    inline void ftrl_UpdateV(float* z_v, float *w, float* cg, float const* g, int n,
                                  unsigned minibatch_occurence, bool* active) {
        for (int i = 0; i < n; i++) {
            float grad = g[i];
            float cg_old = cg[i];
            float cg_new = (float) sqrt(cg_old * cg_old + grad * grad);
            z_v[i] += grad - (cg_new - cg_old) / V.alpha * w[i];
            cg[i] = cg_new;
        }

        float l1 = V.lambda_l1 + minibatch_occurence * V.lambda_l1_incremental;
        float l2 = V.lambda_l2 + minibatch_occurence * V.lambda_l2_incremental;
        float l1_2 = V.lambda_l1_2 + minibatch_occurence * V.lambda_l1_2_incremental;


        if (V.lambda_l1_2 > 0 && (!V.l1_2_only_small || n < V.dim + 1)) {
            auto res_w = solve_proximal_operator_group(z_v, cg, l1, l2, l1_2, n, active);
            for (int i = 0; i < n; i++) {
                w[i] = res_w[i];
            }
        } else {
            for (int i = 0; i < n; i++) {
                w[i] = solve_proximal_operator(- z_v[i], V.alpha / (cg[i] + V.beta), l1, l2);
            }
        }
    }

    // FTRL-Proximal with RMSProp learning rates
    inline void ftrl_rmsprop_UpdateV(float* z_v, float *w, float* cg, float const* g, int n,
                                     unsigned minibatch_occurence, float lr_nu_power, bool* active) {
        float *n_t = new float [n];
        for (int i = 0; i < n; i++) {
            float grad = g[i];

            // temporary crutch for initial bias correction term
            float n_t_prev;
            if(lr_nu_power < V.lr_nu) {
                n_t_prev = (float) sqrt((cg[i] / (1.0f - lr_nu_power / V.lr_nu)));
            } else {
                n_t_prev = 0;
            }
            cg[i] = V.lr_nu * cg[i] + (1.0f - V.lr_nu) * grad * grad;
            float n_t_cur = (float) sqrt(cg[i] / (1.0f - lr_nu_power));
            n_t[i] = n_t_cur;
            z_v[i] += grad - (n_t_cur - n_t_prev) / V.alpha * w[i];
        }

        float l1 = V.lambda_l1 + minibatch_occurence * V.lambda_l1_incremental;
        float l2 = V.lambda_l2 + minibatch_occurence * V.lambda_l2_incremental;
        float l1_2 = V.lambda_l1_2 + minibatch_occurence * V.lambda_l1_2_incremental;


        if (V.lambda_l1_2 > 0 && (!V.l1_2_only_small || n < V.dim + 1)) {
            auto res_w = solve_proximal_operator_group(z_v, n_t, l1, l2, l1_2, n, active);
            for (int i = 0; i < n; i++) {
                w[i] = res_w[i];
            }
        } else {
            for (int i = 0; i < n; i++) {
                w[i] = solve_proximal_operator(- z_v[i], V.alpha / (n_t[i] + V.beta), l1, l2);
            }
        }
        delete[] n_t;
    }

    /// I wish my equations are right...
    inline void ftrl_adam_UpdateV(float *m, float *z_v, float *w, float *cg, float const* g, int n,
                                  unsigned minibatch_occurence, float momentum_mu_power,
                                  float lr_nu_power, bool *active)
    {
        float *n_t = new float [n];
        for (int i = 0; i < n; i++) {
            float grad = g[i];

            // temporary crutch for initial bias correction term
            float n_t_prev;
            if(lr_nu_power < V.lr_nu) {
                n_t_prev = (float) sqrt((cg[i] / (1.0f - lr_nu_power / V.lr_nu)));
            } else {
                n_t_prev = 0;
            }
            cg[i] = V.lr_nu * cg[i] + (1.0f - V.lr_nu) * grad * grad;
            float n_t_cur = (float) sqrt(cg[i] / (1.0f - lr_nu_power));
            n_t[i] = n_t_cur;
            /// ADAM! Woohoo!
            m[i] = V.momentum_mu * m[i] + (1 - V.momentum_mu) * grad;
            z_v[i] += m[i] / (1.0f - momentum_mu_power) - (n_t_cur - n_t_prev) / V.alpha * w[i];
        }


        float l1 = V.lambda_l1 + minibatch_occurence * V.lambda_l1_incremental;
        float l2 = V.lambda_l2 + minibatch_occurence * V.lambda_l2_incremental;
        float l1_2 = V.lambda_l1_2 + minibatch_occurence * V.lambda_l1_2_incremental;


        if (V.lambda_l1_2 > 0 && (!V.l1_2_only_small || n < V.dim + 1)) {
            auto res_w = solve_proximal_operator_group(z_v, n_t, l1, l2, l1_2, n, active);
            for (int i = 0; i < n; i++) {
                w[i] = res_w[i];
            }
        } else {
            for (int i = 0; i < n; i++) {
                w[i] = solve_proximal_operator(- z_v[i], V.alpha / (n_t[i] + V.beta), l1, l2);
            }
        }
        delete[] n_t;
    }


    inline void ftrl_nadam_UpdateV(float *m, float *z_v, float *w, float *cg, float const* g, int n,
                                  unsigned minibatch_occurence, float momentum_mu_power,
                                  float lr_nu_power, bool *active)
    {
        float *n_t = new float [n];
        for (int i = 0; i < n; i++) {
            float grad = g[i];
            // Subtracting nesterov momentum here
            z_v[i] -= V.momentum_mu * m[i] / (1.0f - momentum_mu_power) * (1 - V.momentum_mu);

            // temporary crutch for initial bias correction term
            float n_t_prev;
            if(lr_nu_power < V.lr_nu) {
                n_t_prev = (float) sqrt((cg[i] / (1.0f - lr_nu_power / V.lr_nu)));
            } else {
                n_t_prev = 0;
            }
            cg[i] = V.lr_nu * cg[i] + (1.0f - V.lr_nu) * grad * grad;
            float n_t_cur = (float) sqrt(cg[i] / (1.0f - lr_nu_power));
            n_t[i] = n_t_cur;
            /// ADAM! Woohoo!
            m[i] = V.momentum_mu * m[i] + (1 - V.momentum_mu) * grad;
            z_v[i] += m[i] / (1.0f - momentum_mu_power) - (n_t_cur - n_t_prev) / V.alpha * w[i];

            // Incorporating nesterov momentum here
            z_v[i] += V.momentum_mu * m[i] / (1.0f - momentum_mu_power * V.momentum_mu) * (1 - V.momentum_mu);
        }


        float l1 = V.lambda_l1 + minibatch_occurence * V.lambda_l1_incremental;
        float l2 = V.lambda_l2 + minibatch_occurence * V.lambda_l2_incremental;
        float l1_2 = V.lambda_l1_2 + minibatch_occurence * V.lambda_l1_2_incremental;


        if (V.lambda_l1_2 > 0 && (!V.l1_2_only_small || n < V.dim + 1)) {
            auto res_w = solve_proximal_operator_group(z_v, n_t, l1, l2, l1_2, n, active);
            for (int i = 0; i < n; i++) {
                w[i] = res_w[i];
            }
        } else {
            for (int i = 0; i < n; i++) {
                w[i] = solve_proximal_operator(- z_v[i], V.alpha / (n_t[i] + V.beta), l1, l2);
            }
        }
        delete[] n_t;
    }
};

class AsyncServer : public solver::MinibatchServer {
public:
    AsyncServer(const Config& conf) : conf_(conf) {
        using Server = ps::OnlineServer<float, AdaGradEntry, AdaGradHandle>;
        AdaGradHandle h;
        h.reporter = [this](const Progress& prog) { ReportToScheduler(prog.data); };

        // for w
        h.alpha         = conf.lr_eta();
        h.beta          = conf.lr_beta();
        h.lambda_l1     = conf.lambda_l1();
        h.lambda_l2     = conf.lambda_l2();
        h.l1_shrk       = conf.l1_shrk();
        h.learn_bias_embedding = conf.learn_bias_embedding();
        h.algo_w        = conf.algo_w();

        // for V
        if (conf.embedding_size() > 0) {
            const auto& c = conf.embedding(0);
            h.V.dim         = c.dim();
            h.V.thr         = (unsigned)c.threshold();
            h.V.thr_step    = (unsigned)c.threshold_step();
            h.V.lambda_l2   = c.lambda_l2();
            h.V.lambda_l1   = c.lambda_l1();
            h.V.V_min       = - c.init_scale();
            h.V.V_max       = c.init_scale();
            h.V.alpha       = c.has_lr_eta() ? c.lr_eta() : h.alpha;
            h.V.beta        = c.has_lr_beta() ? c.lr_beta() : h.beta;
            h.V.algo_v      = c.algo_v();
            h.V.lambda_l1_2 = c.lambda_l1_2();
            h.V.lr_nu       = c.lr_nu();
            h.V.momentum_mu = c.momentum_mu();
            h.V.l1_2_only_small = c.l1_2_only_small();

            h.V.lambda_l2_incremental = c.lambda_l2_incremental();
            h.V.lambda_l1_incremental = c.lambda_l1_incremental();
            h.V.lambda_l1_2_incremental = c.lambda_l1_2_incremental();
        }

        Server s(h, 1, 1, ps::NextID(), conf_.max_keys());
        server_ = s.server();
    }

    virtual ~AsyncServer() { }
protected:
    virtual void LoadModel(Stream* fi, bool full_state_mode = false) {
        LOG(INFO) << "Trying to load model; full_state = " << full_state_mode;
        server_->Load(fi, full_state_mode);

        Progress prog;
        prog.new_w() = ISGDHandle::new_w; prog.new_V() = ISGDHandle::new_V;
        ReportToScheduler(prog.data);
    }

    virtual void SaveModel(Stream* fo, bool full_state_mode = false) const {
        LOG(INFO) << "Trying to save model; full_state = " << full_state_mode;
        server_->Save(fo, full_state_mode);
    }
    ps::KVStore* server_;
    Config conf_;
};

class AsyncWorker : public solver::MinibatchWorker {
public:
    AsyncWorker(const Config& conf) : conf_(conf) {
        mb_size_       = conf_.minibatch();
        shuffle_       = conf_.rand_shuffle();
        concurrent_mb_ = conf_.max_concurrency();
        neg_sampling_  = conf_.neg_sampling();

        for (int i = 0; i < conf.embedding_size(); ++i) {
            if (conf.embedding(i).dim() > 0) {
                do_embedding_ = true; break;
            }
        }

        // get learn namespaces from ml-engine
        learn_namespaces_.resize(conf.learn_namespaces_size());
        int ns_idx = 0;
        for (const auto & ns_set: conf.learn_namespaces()) {
            for (const auto & ns: ns_set.namespace_idxs()) {
                learn_namespaces_[ns_idx].push_back(ns);
            }
            ns_idx++;
        }
    }
    virtual ~AsyncWorker() { }

protected:
    virtual void ProcessMinibatch(const Minibatch& mb, const Workload& wl) {
        auto data = new dmlc::data::RowBlockContainer<unsigned>();
        auto feaid = std::make_shared <std::vector<FeaID>>();
        auto feacnt = std::make_shared <std::vector<float>>();

        double start = GetTime();
        Localizer<FeaID> lc(conf_.num_threads());
        lc.Localize(mb, data, feaid.get(), feacnt.get());
        workload_time_ += GetTime() - start;

        ps::SyncOpts pull_w_opt;
        if (wl.type == Workload::TRAIN && wl.data_pass == 0 && do_embedding_) {
            // push the feature count to the servers
            ps::SyncOpts cnt_opt;
            SetFilters(0, &cnt_opt);
            cnt_opt.cmd = kPushFeaCnt;
            int t = server_.ZPush(feaid, feacnt, cnt_opt);
            pull_w_opt.deps.push_back(t);
            // LL << DebugStr(*feacnt);
        }

        // pull the weight from the servers
        auto val = new std::vector<float>();
        auto val_siz = new std::vector<int>();

        // this callback will be called when the weight has been actually pulled
        // back
        pull_w_opt.callback = [this, data, feaid, val, val_siz, wl]() {
            double start = GetTime();
            // eval the objective, and report progress to the scheduler
            Loss<float> loss(data->GetBlock(), *val, *val_siz, conf_);
            Progress prog; loss.Evaluate(&prog); ReportToScheduler(prog.data);
            if (wl.type == Workload::PRED) {
                loss.Predict(PredictStream(conf_.predict_out(), wl), conf_.prob_predict());
            } else if (wl.type == Workload::TRAIN) {
                // calculate and push the gradient
                loss.CalcGrad(val);

                ps::SyncOpts push_grad_opt;
                // filters to reduce network traffic
                SetFilters(2, &push_grad_opt);
                // this callback will be called when the gradients have been actually
                // pushed
                // LL << DebugStr(*val);
                push_grad_opt.callback = [this]() { FinishMinibatch(); };
                server_.ZVPush(feaid,
                               std::shared_ptr<std::vector<float>>(val),
                               std::shared_ptr<std::vector<int>>(val_siz),
                               push_grad_opt);

            } else {
                FinishMinibatch();
                delete val;
                delete val_siz;
            }
            delete data;
            workload_time_ += GetTime() - start;
        };
        // filters to reduce network traffic
        SetFilters(1, &pull_w_opt);
        server_.ZVPull(feaid, val, val_siz, pull_w_opt);
    }

private:
    // flag: 0 push feature count, 1 pull weight, 2 push gradient
    void SetFilters(int flag, ps::SyncOpts* opts) {
        if (conf_.key_cache()) {
            opts->AddFilter(ps::Filter::KEY_CACHING)->set_clear_cache(flag == 2);
        }
        if (conf_.fixed_bytes() > 0) {
            if (flag == 0) {
                // trancate the count to uint8
                opts->AddFilter(ps::Filter::TRUNCATE_FLOAT)->set_num_bytes(1);
            } else {
                // randomly round the gradient
                opts->AddFilter(ps::Filter::FIXING_FLOAT)->set_num_bytes(
                        conf_.fixed_bytes());
            }
        }
        if (conf_.msg_compression()) {
            opts->AddFilter(ps::Filter::COMPRESSING);
        }
    }

    Config conf_;
    bool do_embedding_ = false;
    ps::KVWorker<float> server_;
};


}  // namespace difacto
}  // namespace dmlc
