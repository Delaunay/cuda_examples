MACRO(CUDA_EXEC NAME SRC)
    add_executable(${NAME} ${SRC})
    target_link_libraries(${NAME} cuda ${CUDA_LIBRARIES})
ENDMACRO(CUDA_EXEC)

CUDA_EXEC(vectorAdd vectorAdd/vectorAdd.cu)

CUDA_EXEC(vectorAddDrv
    vectorAddDrv/vectorAddDrv.cpp
    vectorAddDrv/vectorAdd_kernel.cu)
