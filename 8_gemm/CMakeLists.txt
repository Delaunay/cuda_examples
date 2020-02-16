MACRO(CUDA_EXEC NAME SRC)
    add_executable(${NAME} ${SRC})
    target_link_libraries(${NAME} cuda ${CUDA_LIBRARIES})
ENDMACRO(CUDA_EXEC)

CUDA_EXEC(bad_gemm bad_gemm.cu)
CUDA_EXEC(better_gemm better_gemm.cu)

