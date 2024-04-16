#include <unordered_map>
#include <vector>

namespace falconn {
namespace core {
  class MetadataStore {
    public:
      MetadataStore(std::unordered_map<int, std::vector<int>> small_labels) : small_labels_(small_labels) {}
      ~MetadataStore() {}

      std::vector<int> get_indices_for_label(int label) {
        return small_labels_[label];
      }

      std::unordered_map<int, std::vector<int>>& get_small_labels() {
        return small_labels_;
      }
      private:
        std::unordered_map<int, std::vector<int>> small_labels_;
  };
}  // namespace core
}  // namespace falconn
