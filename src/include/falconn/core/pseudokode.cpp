find_nearest_neighbour(query, num_probes, metadata_filter) {
  const cands = find_candidates()

  best_distance = fucking en million
  best_point = nullptr
  for cand in cands {
    if (GLOBAL_METADATA_TABLE.lookup(cand).contains(metadata_filter)) {
      const dist = check_distance(cand, query)
      if dist < best_distance {
        best_distance = dist
        best_point = cand
      }
    }
  }
}