#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    // Round i (0-indexed in loop) processes keys[0..i] and values[0..i]
    // current_query has shape [i+1, d] where d=512
    size_t num_kv = i + 1;
    size_t d = 512;

    // Step 1: Concatenate all keys into K_all (shape [i+1, d])
    // Each key[j] is [1, d], concat along axis 0 to get [i+1, d]
    Matrix *K_all = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (K_all == nullptr) {
        // First key, copy it
        K_all = matrix_memory_allocator.Allocate("K_all");
        gpu_sim.Copy(keys[j], K_all, Position::kInGpuHbm);
      } else {
        // Concatenate with previous keys
        Matrix *new_K_all = matrix_memory_allocator.Allocate("K_all_new");
        gpu_sim.Concat(K_all, keys[j], new_K_all, 0, Position::kInGpuHbm);
        gpu_sim.ReleaseMatrix(K_all);
        K_all = new_K_all;
      }
    }

    // Step 2: Compute Q * K^T
    // Q: [i+1, d], K_all: [i+1, d], K_all^T: [d, i+1]
    // Q * K^T: [i+1, d] * [d, i+1] = [i+1, i+1]
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K_all);
    gpu_sim.Transpose(K_all, Position::kInSharedMemory);

    Matrix *QK = matrix_memory_allocator.Allocate("QK");
    gpu_sim.MatMul(current_query, K_all, QK);

    // Step 3: Apply row-wise softmax on QK
    // QK is [i+1, i+1], need to apply softmax to each row
    Matrix *QK_exp = matrix_memory_allocator.Allocate("QK_exp");
    gpu_sim.MatExp(QK, QK_exp);
    gpu_sim.ReleaseMatrix(QK);

    // Now need to compute row sums and divide each row by its sum
    // Reshape QK_exp to process more easily
    // QK_exp: [i+1, i+1]

    Matrix *softmax_QK = matrix_memory_allocator.Allocate("softmax_QK");

    // For each row i, we need to:
    // 1. Get row i
    // 2. Compute sum of that row
    // 3. Divide each element by the sum

    Matrix *first_row = nullptr;
    for (size_t row_idx = 0; row_idx < num_kv; ++row_idx) {
      // Get row
      Matrix *row = matrix_memory_allocator.Allocate("row_" + std::to_string(row_idx));
      gpu_sim.GetRow(QK_exp, row_idx, row, Position::kInSharedMemory);

      // Compute sum of this row
      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(row_idx));
      gpu_sim.Sum(row, row_sum);

      // Divide row by sum
      Matrix *normalized_row = matrix_memory_allocator.Allocate("norm_row_" + std::to_string(row_idx));
      gpu_sim.MatDiv(row, row_sum, normalized_row);

      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_sum);

      // Concatenate rows to build softmax_QK
      if (first_row == nullptr) {
        first_row = normalized_row;
      } else {
        Matrix *new_softmax = matrix_memory_allocator.Allocate("softmax_concat");
        gpu_sim.MoveMatrixToGpuHbm(first_row);
        gpu_sim.MoveMatrixToGpuHbm(normalized_row);
        gpu_sim.Concat(first_row, normalized_row, new_softmax, 0, Position::kInGpuHbm);
        gpu_sim.ReleaseMatrix(first_row);
        gpu_sim.ReleaseMatrix(normalized_row);
        first_row = new_softmax;
      }
    }

    softmax_QK = first_row;
    gpu_sim.ReleaseMatrix(QK_exp);

    // Step 4: Concatenate all values into V_all
    Matrix *V_all = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (V_all == nullptr) {
        V_all = matrix_memory_allocator.Allocate("V_all");
        gpu_sim.Copy(values[j], V_all, Position::kInGpuHbm);
      } else {
        Matrix *new_V_all = matrix_memory_allocator.Allocate("V_all_new");
        gpu_sim.Concat(V_all, values[j], new_V_all, 0, Position::kInGpuHbm);
        gpu_sim.ReleaseMatrix(V_all);
        V_all = new_V_all;
      }
    }

    // Step 5: Result = softmax(QK) * V_all
    gpu_sim.MoveMatrixToSharedMem(softmax_QK);
    gpu_sim.MoveMatrixToSharedMem(V_all);

    Matrix *result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(softmax_QK, V_all, result);

    gpu_sim.MoveMatrixToGpuHbm(result);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu