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
        RT_RET_ERROR_VERSION_UNMATCH = rt_return_value_t::RT_RET_ERROR_VERSION_UNMATCH,
        RT_RET_ERROR_ALLOCATE_CONTEXT = rt_return_value_t::RT_RET_ERROR_ALLOCATE_CONTEXT,
        RT_RET_ERROR_INITIALIZE_CONTEXT_TWICE = rt_return_value_t::RT_RET_ERROR_INITIALIZE_CONTEXT_TWICE,
        RT_RET_ERROR_ALLOCATE_CALLBACK_BUFFER = rt_return_value_t::RT_RET_ERROR_ALLOCATE_CALLBACK_BUFFER,
        RT_RET_ERROR_INVALID_BUFFER_INDEX = rt_return_value_t::RT_RET_ERROR_INVALID_BUFFER_INDEX,
        RT_RET_ERROR_INIT_VARIABLE = rt_return_value_t::RT_RET_ERROR_INIT_VARIABLE,
        RT_RET_ERROR_UNKNOWN_FUNCTION = rt_return_value_t::RT_RET_ERROR_UNKNOWN_FUNCTION,
        RT_RET_ERROR_NO_MATCHING_FUNCTION = rt_return_value_t::RT_RET_ERROR_NO_MATCHING_FUNCTION,
        RT_RET_NOERROR = rt_return_value_t::RT_RET_NOERROR,
        RT_RET_FUNCTION_MATCH = rt_return_value_t::RT_RET_FUNCTION_MATCH,
        RT_RET_FUNCTION_DONT_MATCH = rt_return_value_t::RT_RET_FUNCTION_DONT_MATCH,
        RT_RET_END_OF_VALUES = rt_return_value_t::RT_RET_END_OF_VALUES
    };

    enum DataType {
        NN_DATA_TYPE_FLOAT = nn_data_type_t::NN_DATA_TYPE_FLOAT,   ///< 32bit float.
        NN_DATA_TYPE_INT16 = nn_data_type_t::NN_DATA_TYPE_INT16,   ///< 16bit integer.
        NN_DATA_TYPE_INT8 = nn_data_type_t::NN_DATA_TYPE_INT8,     ///<  8bit integer.
        NN_DATA_TYPE_SIGN = nn_data_type_t::NN_DATA_TYPE_SIGN,     ///< Binary.
        END_OF_NN_DATA_TYPE = nn_data_type_t::END_OF_NN_DATA_TYPE
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
