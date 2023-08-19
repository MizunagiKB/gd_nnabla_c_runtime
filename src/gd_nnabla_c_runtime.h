#ifndef GD_NNABLA_C_RUNTIME_H
#define GD_NNABLA_C_RUNTIME_H

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/class_db.hpp>

#include <nnablart/network.h>
#include <nnablart/runtime.h>

using namespace godot;

class GDNNablaCRuntime : public RefCounted {
    GDCLASS(GDNNablaCRuntime, RefCounted);

protected:
    rt_context_pointer context = 0;
    nn_network_t *net = nullptr;

public:
    enum ReturnValue {
        ERROR_VERSION_UNMATCH = RT_RET_ERROR_VERSION_UNMATCH,
        ERROR_ALLOCATE_CONTEXT = RT_RET_ERROR_ALLOCATE_CONTEXT,
        ERROR_INITIALIZE_CONTEXT_TWICE = RT_RET_ERROR_INITIALIZE_CONTEXT_TWICE,
        ERROR_ALLOCATE_CALLBACK_BUFFER = RT_RET_ERROR_ALLOCATE_CALLBACK_BUFFER,
        ERROR_INVALID_BUFFER_INDEX = RT_RET_ERROR_INVALID_BUFFER_INDEX,
        ERROR_INIT_VARIABLE = RT_RET_ERROR_INIT_VARIABLE,
        ERROR_UNKNOWN_FUNCTION = RT_RET_ERROR_UNKNOWN_FUNCTION,
        ERROR_NO_MATCHING_FUNCTION = RT_RET_ERROR_NO_MATCHING_FUNCTION,
        NOERROR = RT_RET_NOERROR,
        FUNCTION_MATCH = RT_RET_FUNCTION_MATCH,
        FUNCTION_DONT_MATCH = RT_RET_FUNCTION_DONT_MATCH,
        END_OF_VALUES = RT_RET_END_OF_VALUES
    };

    enum DataType {
        DATA_TYPE_FLOAT = NN_DATA_TYPE_FLOAT,   ///< 32bit float.
        DATA_TYPE_INT16 = NN_DATA_TYPE_INT16,   ///< 16bit integer.
        DATA_TYPE_INT8 = NN_DATA_TYPE_INT8,     ///<  8bit integer.
        DATA_TYPE_SIGN = NN_DATA_TYPE_SIGN,     ///< Binary.
        END_OF_DATA_TYPE = END_OF_NN_DATA_TYPE
    };

protected:
    static void _bind_methods();

public:
    ReturnValue rt_allocate_context();
    ReturnValue rt_initialize_context(PackedByteArray nnb);
    ReturnValue rt_free_context();

    int rt_num_of_input() const;
    int rt_input_size(int p_idx) const;
    int rt_input_dimension(int p_idx) const;
    int rt_input_shape(int p_idx, int p_shape_idx) const;
    bool rt_input_buffer(int p_idx, PackedFloat32Array input);

    int rt_num_of_output() const;
    int rt_output_size(int p_idx) const;
    int rt_output_dimension(int index) const;
    int rt_output_shape(int index, int p_shape_idx) const;
    PackedFloat32Array rt_output_buffer(int p_idx);

    Dictionary rt_input_variable(int p_idx);
    Dictionary rt_output_variable(int p_idx);

    ReturnValue rt_forward();

    String rt_nnabla_version() const;
    String rt_c_runtime_version() const;
    int rt_nnb_version() const;
    String rt_nnb_revision() const;

    GDNNablaCRuntime();
    ~GDNNablaCRuntime();
};

VARIANT_ENUM_CAST(GDNNablaCRuntime::ReturnValue);
VARIANT_ENUM_CAST(GDNNablaCRuntime::DataType);

#endif // GD_NNABLA_C_RUNTIME_H
