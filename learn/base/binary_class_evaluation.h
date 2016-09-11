#pragma once
#include <algorithm>
#include <dmlc/logging.h>
#include <dmlc/omp.h>
namespace dmlc {

template <typename V>
class BinClassEval {
 public:
  BinClassEval(const V* const label,
             const V* const predict,
             size_t n,
             int num_threads = 2)
      : label_(label), predict_(predict), size_(n), nt_(num_threads) { }
  ~BinClassEval() { }

  V AUC() {
    size_t n = size_;
    struct Entry { V label; V predict; };
    std::vector<Entry> buff(n);
    for (size_t i = 0; i < n; ++i) {
      buff[i].label = label_[i];
      buff[i].predict = predict_[i];
    }
    std::sort(buff.data(), buff.data()+n,  [](const Entry& a, const Entry&b) {
        return a.predict < b.predict; });
    V area = 0, cum_tp = 0;
    for (size_t i = 0; i < n; ++i) {
      if (buff[i].label > 0) {
        cum_tp += 1;
      } else {
        area += cum_tp;
      }
    }
    if (cum_tp == 0 || cum_tp == n) return 1;
    area /= cum_tp * (n - cum_tp);
    return area < 0.5 ? 1 - area : area;
  }

  // In the future, we will make one AUC function returning all AUCs to avoid double sorting
  V AUC_PR() {
    size_t n = size_;
    struct Entry { V label; V predict; };
    std::vector<Entry> buff(n);
    for (size_t i = 0; i < n; ++i) {
      buff[i].label = label_[i];
      buff[i].predict = predict_[i];
    }
    //reverse sort
    std::sort(buff.data(), buff.data()+n,  [](const Entry& a, const Entry&b) {
        return a.predict > b.predict; });
    size_t true_predicted_labels = 0;
    V cumulative_precision = 0.0;
    for(size_t i = 0; i < n; ++i) {
      if (buff[i].label > 0) {
        true_predicted_labels++;
        cumulative_precision += (V)true_predicted_labels / (i + 1);
      }
    }
    return cumulative_precision / true_predicted_labels;
  }

  std::vector <V> AUC_all() {
    size_t n = size_;
    struct Entry { V label; V predict; };
    std::vector<Entry> buff(n);
    for (size_t i = 0; i < n; ++i) {
      buff[i].label = label_[i];
      buff[i].predict = predict_[i];
    }
    std::sort(buff.data(), buff.data()+n,  [](const Entry& a, const Entry&b) {
        return a.predict < b.predict; });
    V area = 0, cum_tp = 0;
    for (size_t i = 0; i < n; ++i) {
      if (buff[i].label > 0) {
        cum_tp += 1;
      } else {
        area += cum_tp;
      }
    }
    if (cum_tp == 0 || cum_tp == n) area = 1;
    area /= cum_tp * (n - cum_tp);
    area = area < 0.5 ? 1 - area : area;
    
    
    size_t true_predicted_labels = 0;
    V cumulative_precision = 0.0;
    for(size_t i = n; i > 0; --i) {
      if (buff[i - 1].label > 0) {
        true_predicted_labels++;
        cumulative_precision += (V)true_predicted_labels / i;
      }
    }
    V area_pr = cumulative_precision / true_predicted_labels;
    return std::vector <V> ({area, area_pr});
  }

  V Precision(V threshold) {
    V correct = 0;
    size_t n = size_;
#pragma omp parallel for reduction(+:correct,y_predicted_count) num_threads(nt_)
    V y_predicted_count = 1;
    for (size_t i = 0; i < n; ++i) {
      if (predict_[i] > threshold) {
        y_predicted_count += 1;
        if(label_[i] > 0) {
          correct += 1;
        }
      }
    }
    return correct / y_predicted_count;
  }

  V Recall(V threshold) {
    V correct = 0;
    size_t n = size_;
#pragma omp parallel for reduction(+:correct, y_1_count) num_threads(nt_)
    V y_1_count = 1;
    for (size_t i = 0; i < n; ++i) {
      if (label_[i] > 0) {
        y_1_count += 1;
        if (predict_[i] > threshold) {
          correct += 1;
        }
      }
    }
    return correct / y_1_count;
  }


  V Accuracy(V threshold) {
    V correct = 0;
    size_t n = size_;
#pragma omp parallel for reduction(+:correct) num_threads(nt_)
    for (size_t i = 0; i < n; ++i) {
      if ((label_[i] > 0 && predict_[i] > threshold) ||
          (label_[i] <= 0 && predict_[i] <= threshold))
        correct += 1;
    }
    V acc = correct / (V) n;
    return acc > 0.5 ? acc : 1 - acc;
  }

  V LogLoss() {
    V loss = 0;
    size_t n = size_;
#pragma omp parallel for reduction(+:loss) num_threads(nt_)
    for (size_t i = 0; i < n; ++i) {
      V y = label_[i] > 0;
      V p = 1 / (1 + exp(- predict_[i]));
      p = p < 1e-10 ? 1e-10 : p;
      loss += y * log(p) + (1 - y) * log(1 - p);
    }
    return - loss;
  }

  V LogitObjv() {
    V objv = 0;
#pragma omp parallel for reduction(+:objv) num_threads(nt_)
    for (size_t i = 0; i < size_; ++i) {
      V y = label_[i] > 0 ? 1 : -1;
      objv += log( 1 + exp( - y * predict_[i] ));
    }
    return objv;
  }
  
  V Copc(){
    V clk = 0;
    V clk_exp = 0.0;
#pragma omp parallel for reduction(+:clk,clk_exp) num_threads(nt_)
    for (size_t i = 0; i < size_; ++i) {
      if (label_[i] > 0) clk += 1;
      clk_exp += 1.0 / ( 1.0 + exp( - predict_[i] ));
    }
    return clk / clk_exp;
  }

 private:
  V const* label_;
  V const* predict_;
  size_t size_;
  int nt_;
};


}  // namespace dmlc
