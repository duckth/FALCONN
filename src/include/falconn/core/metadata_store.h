#include <unordered_map>
#include <unordered_set>

namespace falconn {
namespace core {
  const std::unordered_map<int, std::unordered_set<int>> small_labels =
      {
        // Insert data here
      };
  class MetadataStore {
    public:
      MetadataStore() : small_labels_(small_labels) {}
      ~MetadataStore() {}

      std::unordered_set<int> get_indices_for_label(int label) {
        return small_labels_[label];
      }

      const std::unordered_map<int, std::unordered_set<int>>& get_small_labels() const {
        return small_labels_;
      }
      private:
        const std::unordered_map<int, std::unordered_set<int>> small_labels_;
  };
}  // namespace core
}  // namespace falconn