#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    size_t num_kv = i + 1;
    size_t d = 512;

    // Step 1: Build K_all by concatenating keys in SRAM
    Matrix *K_all = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      Matrix *key_copy = matrix_memory_allocator.Allocate("key_copy");
      gpu_sim.Copy(keys[j], key_copy, Position::kInGpuHbm);
      gpu_sim.MoveMatrixToSharedMem(key_copy);

      if (K_all == nullptr) {
        K_all = key_copy;
      } else {
        Matrix *new_K_all = matrix_memory_allocator.Allocate("K_all_new");
        gpu_sim.Concat(K_all, key_copy, new_K_all, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(K_all);
        gpu_sim.ReleaseMatrix(key_copy);
        K_all = new_K_all;
      }
    }

    // Step 2: Compute Q * K^T in SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.Transpose(K_all, Position::kInSharedMemory);

    Matrix *QK = matrix_memory_allocator.Allocate("QK");
    gpu_sim.MatMul(current_query, K_all, QK);
    gpu_sim.ReleaseMatrix(K_all);

    // Step 3: Row-wise softmax in SRAM
    Matrix *QK_exp = matrix_memory_allocator.Allocate("QK_exp");
    gpu_sim.MatExp(QK, QK_exp);
    gpu_sim.ReleaseMatrix(QK);

    // Build softmax result row by row in SRAM
    Matrix *softmax_QK = nullptr;
    for (size_t row_idx = 0; row_idx < num_kv; ++row_idx) {
      Matrix *row = matrix_memory_allocator.Allocate("row");
      gpu_sim.GetRow(QK_exp, row_idx, row, Position::kInSharedMemory);

      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row, row_sum);

      Matrix *normalized_row = matrix_memory_allocator.Allocate("norm_row");
      gpu_sim.MatDiv(row, row_sum, normalized_row);

      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_sum);

      if (softmax_QK == nullptr) {
        softmax_QK = normalized_row;
      } else {
        Matrix *new_softmax = matrix_memory_allocator.Allocate("softmax_new");
        gpu_sim.Concat(softmax_QK, normalized_row, new_softmax, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_QK);
        gpu_sim.ReleaseMatrix(normalized_row);
        softmax_QK = new_softmax;
      }
    }
    gpu_sim.ReleaseMatrix(QK_exp);

    // Step 4: Build V_all in SRAM
    Matrix *V_all = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      Matrix *value_copy = matrix_memory_allocator.Allocate("value_copy");
      gpu_sim.Copy(values[j], value_copy, Position::kInGpuHbm);
      gpu_sim.MoveMatrixToSharedMem(value_copy);

      if (V_all == nullptr) {
        V_all = value_copy;
      } else {
        Matrix *new_V_all = matrix_memory_allocator.Allocate("V_all_new");
        gpu_sim.Concat(V_all, value_copy, new_V_all, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(V_all);
        gpu_sim.ReleaseMatrix(value_copy);
        V_all = new_V_all;
      }
    }

    // Step 5: Final multiplication in SRAM
    Matrix *result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(softmax_QK, V_all, result);

    gpu_sim.ReleaseMatrix(softmax_QK);
    gpu_sim.ReleaseMatrix(V_all);
    gpu_sim.MoveMatrixToGpuHbm(current_query);
    gpu_sim.MoveMatrixToGpuHbm(result);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
