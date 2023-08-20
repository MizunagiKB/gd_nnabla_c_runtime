#include "gd_nnabla_c_runtime.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;


void *gd_rt_variable_malloc_func(size_t size) {
    return memalloc(size);
}

void gd_rt_variable_free_func(void *ptr) {
    memfree(ptr);
}

void *gd_rt_malloc_func(size_t size) {
    return memalloc(size);
}

void gd_rt_free_func(void *ptr) {
    memfree(ptr);
}

void GDNNablaCRuntime::_bind_methods() {
    ClassDB::bind_method(D_METHOD("rt_allocate_context"), &GDNNablaCRuntime::rt_allocate_context);
    ClassDB::bind_method(D_METHOD("rt_initialize_context", "nnb"), &GDNNablaCRuntime::rt_initialize_context);
    ClassDB::bind_method(D_METHOD("rt_free_context"), &GDNNablaCRuntime::rt_free_context);

    ClassDB::bind_method(D_METHOD("rt_num_of_input"), &GDNNablaCRuntime::rt_num_of_input);
    ClassDB::bind_method(D_METHOD("rt_input_size", "idx"), &GDNNablaCRuntime::rt_input_size);
    ClassDB::bind_method(D_METHOD("rt_input_dimension", "idx"), &GDNNablaCRuntime::rt_input_dimension);
    ClassDB::bind_method(D_METHOD("rt_input_shape", "idx", "shape_idx"), &GDNNablaCRuntime::rt_input_shape);
    ClassDB::bind_method(D_METHOD("rt_input_buffer", "idx", "input"), &GDNNablaCRuntime::rt_input_buffer);

    ClassDB::bind_method(D_METHOD("rt_num_of_output"), &GDNNablaCRuntime::rt_num_of_output);
    ClassDB::bind_method(D_METHOD("rt_output_size"), &GDNNablaCRuntime::rt_output_size);
    ClassDB::bind_method(D_METHOD("rt_output_dimension", "idx"), &GDNNablaCRuntime::rt_output_dimension);
    ClassDB::bind_method(D_METHOD("rt_output_shape", "idx", "shape_idx"), &GDNNablaCRuntime::rt_output_shape);
    ClassDB::bind_method(D_METHOD("rt_output_buffer", "idx"), &GDNNablaCRuntime::rt_output_buffer);

    ClassDB::bind_method(D_METHOD("rt_input_variable", "idx"), &GDNNablaCRuntime::rt_input_variable);
    ClassDB::bind_method(D_METHOD("rt_output_variable", "idx"), &GDNNablaCRuntime::rt_output_variable);

    ClassDB::bind_method(D_METHOD("rt_forward"), &GDNNablaCRuntime::rt_forward);

    ClassDB::bind_method(D_METHOD("rt_nnabla_version"), &GDNNablaCRuntime::rt_nnabla_version);
    ClassDB::bind_method(D_METHOD("rt_c_runtime_version"), &GDNNablaCRuntime::rt_c_runtime_version);
    ClassDB::bind_method(D_METHOD("rt_nnb_version"), &GDNNablaCRuntime::rt_nnb_version);
    ClassDB::bind_method(D_METHOD("rt_nnb_revision"), &GDNNablaCRuntime::rt_nnb_revision);

    BIND_ENUM_CONSTANT(ERROR_VERSION_UNMATCH);
	BIND_ENUM_CONSTANT(ERROR_ALLOCATE_CONTEXT);
	BIND_ENUM_CONSTANT(ERROR_INITIALIZE_CONTEXT_TWICE);
	BIND_ENUM_CONSTANT(ERROR_ALLOCATE_CALLBACK_BUFFER);
	BIND_ENUM_CONSTANT(ERROR_INVALID_BUFFER_INDEX);
	BIND_ENUM_CONSTANT(ERROR_INIT_VARIABLE);
	BIND_ENUM_CONSTANT(ERROR_UNKNOWN_FUNCTION);
	BIND_ENUM_CONSTANT(ERROR_NO_MATCHING_FUNCTION);
	BIND_ENUM_CONSTANT(NOERROR);
	BIND_ENUM_CONSTANT(FUNCTION_MATCH);
	BIND_ENUM_CONSTANT(FUNCTION_DONT_MATCH);
	BIND_ENUM_CONSTANT(END_OF_VALUES);

    rt_variable_malloc_func = gd_rt_variable_malloc_func;
    rt_variable_free_func = gd_rt_variable_free_func;
    rt_malloc_func = gd_rt_malloc_func;
    rt_free_func = gd_rt_free_func;
}

GDNNablaCRuntime::ReturnValue GDNNablaCRuntime::rt_allocate_context() {
    ERR_FAIL_COND_V(this->context != 0, NOERROR);
    return static_cast<ReturnValue>(::rt_allocate_context(&this->context));
}

GDNNablaCRuntime::ReturnValue GDNNablaCRuntime::rt_initialize_context(PackedByteArray nnb) {
    ERR_FAIL_COND_V(this->context == 0, NOERROR);
    ERR_FAIL_COND_V(this->net != nullptr, NOERROR);

    this->net = static_cast<nn_network_t*>(memalloc(nnb.size()));
    ::memcpy(this->net, nnb.ptr(), nnb.size());

    return static_cast<ReturnValue>(::rt_initialize_context(this->context, this->net));
}

GDNNablaCRuntime::ReturnValue GDNNablaCRuntime::rt_free_context() {
    if(this->context == 0) {
        return ReturnValue::NOERROR;
    }

    if(this->net != nullptr) {
        memfree(this->net);
        this->net = nullptr;
    }

    ReturnValue ret = static_cast<ReturnValue>(::rt_free_context(&this->context));
    this->context = 0;

    return ret;
}

int GDNNablaCRuntime::rt_num_of_input() const {
    return ::rt_num_of_input(this->context);
}

int GDNNablaCRuntime::rt_input_size(int p_idx) const {
    ERR_FAIL_INDEX_V(p_idx, ::rt_num_of_input(this->context), -1);
    return ::rt_input_size(this->context, p_idx);
}

int GDNNablaCRuntime::rt_input_dimension(int p_idx) const {
    ERR_FAIL_INDEX_V(p_idx, ::rt_num_of_input(this->context), -1);
    return ::rt_input_dimension(this->context, p_idx);
}

int GDNNablaCRuntime::rt_input_shape(int p_idx, int p_shape_idx) const {
    ERR_FAIL_INDEX_V(p_idx, ::rt_num_of_input(this->context), -1);
    return ::rt_input_shape(this->context, p_idx, p_shape_idx);
}

bool GDNNablaCRuntime::rt_input_buffer(int p_idx, PackedFloat32Array input) {
    ERR_FAIL_INDEX_V(p_idx, ::rt_num_of_input(this->context), false);
    void* ptr = ::rt_input_buffer(this->context, p_idx);
    memcpy(ptr, input.ptr(), input.size() * sizeof(float));
    return true;
}

int GDNNablaCRuntime::rt_num_of_output() const {
    return ::rt_num_of_output(this->context);
}

int GDNNablaCRuntime::rt_output_size(int p_idx) const {
    ERR_FAIL_INDEX_V(p_idx, ::rt_num_of_output(this->context), -1);
    return ::rt_output_size(this->context, p_idx);
}

int GDNNablaCRuntime::rt_output_dimension(int p_idx) const {
    ERR_FAIL_INDEX_V(p_idx, ::rt_num_of_output(this->context), -1);
    return ::rt_output_dimension(this->context, p_idx);
}

int GDNNablaCRuntime::rt_output_shape(int p_idx, int p_shape_idx) const {
    ERR_FAIL_INDEX_V(p_idx, ::rt_num_of_output(this->context), -1);
    return ::rt_output_shape(this->context, p_idx, p_shape_idx);
}

PackedFloat32Array GDNNablaCRuntime::rt_output_buffer(int p_idx) {
    ERR_FAIL_INDEX_V(p_idx, ::rt_num_of_output(this->context), PackedFloat32Array());

    void* ptr = ::rt_output_buffer(this->context, p_idx);
    int size = ::rt_output_size(this->context, p_idx);

    PackedFloat32Array output;

    output.resize(size);

    memcpy(output.ptrw(), ptr, output.size() * sizeof(float));

    return output;
}

Dictionary GDNNablaCRuntime::rt_input_variable(int p_idx) {
    ERR_FAIL_INDEX_V(p_idx, ::rt_num_of_input(this->context), Dictionary());
    Dictionary result;
    nn_variable_t *variable = ::rt_input_variable(this->context, p_idx);
    result["id"] = variable->id;

    Array shape;
    for(int shape_idx = 0; shape_idx < ::rt_input_dimension(this->context, p_idx); shape_idx++) {
        shape.append(::rt_input_shape(this->context, p_idx, shape_idx));
    }
    result["shape"] = shape;

    result["type"] = variable->type;
    result["fp_pos"] = variable->fp_pos;
    result["data_index"] = variable->data_index;

   return result;
}

Dictionary GDNNablaCRuntime::rt_output_variable(int p_idx) {
    ERR_FAIL_INDEX_V(p_idx, ::rt_num_of_output(this->context), Dictionary());
    Dictionary result;
    nn_variable_t *variable = ::rt_output_variable(this->context, p_idx);
    result["id"] = variable->id;

    Array shape;
    for(int shape_idx = 0; shape_idx < ::rt_output_dimension(this->context, p_idx); shape_idx++) {
        shape.append(::rt_output_shape(this->context, p_idx, shape_idx));
    }
    result["shape"] = shape;

    result["type"] = variable->type;
    result["fp_pos"] = variable->fp_pos;
    result["data_index"] = variable->data_index;

   return result;
}

GDNNablaCRuntime::ReturnValue GDNNablaCRuntime::rt_forward() {
    return static_cast<ReturnValue>(::rt_forward(this->context));
}

String GDNNablaCRuntime::rt_nnabla_version() const {
    return String(::rt_nnabla_version());
}

String GDNNablaCRuntime::rt_c_runtime_version() const {
    return String(::rt_c_runtime_version());
}

int GDNNablaCRuntime::rt_nnb_version() const {
    return ::rt_nnb_version();
}

String GDNNablaCRuntime::rt_nnb_revision() const {
    return String(::rt_nnb_revision());
}

GDNNablaCRuntime::GDNNablaCRuntime() {
}

GDNNablaCRuntime::~GDNNablaCRuntime() {
    this->rt_free_context();
}
