extends Control


var nn_crt_simple: GDNNablaCRuntime;
var list_ui_tex := []
var ary_simple_param := PackedFloat32Array()

var nn_crt_mnist: GDNNablaCRuntime;
var list_ui_edit := []
var list_ui_label := []
var ary_mnist_param: Array[PackedFloat32Array]


func set_texture(tex_rect: TextureRect, idx: int) -> PackedFloat32Array:

    var img = Image.load_from_file("res://addons/gd_nnabla_c_runtime/example/res/t10k/img_%04d.png" % idx)
    var ary_data: PackedFloat32Array

    ary_data.resize(img.get_width() * img.get_height())

    for y in range(img.get_height()):
        for x in range(img.get_width()):
            var c = img.get_pixel(x,y)
            ary_data.set(x + y * img.get_width(), c.r)

    var tex = ImageTexture.create_from_image(img)
    tex_rect.texture = tex

    return ary_data


func get_number(ary_output: Array[float]) -> int:
    var v = ary_output.max()
    return ary_output.find(v)


func init_nnabla(path: String):

    var rf = FileAccess.open(path, FileAccess.READ)
    var rf_size = rf.get_length()
    var net_nnb = rf.get_buffer(rf_size)
    rf.close()
    
    var nn_crt = GDNNablaCRuntime.new()

    var err = nn_crt.rt_allocate_context()
    assert(err == GDNNablaCRuntime.RT_RET_NOERROR)

    nn_crt.rt_initialize_context(net_nnb)

    return nn_crt


func _ready():

    nn_crt_simple = init_nnabla("res://addons/gd_nnabla_c_runtime/example/nn_model/model_simple.nnb")
    nn_crt_mnist = init_nnabla("res://addons/gd_nnabla_c_runtime/example/nn_model/model_mnist.nnb")

    $lbl_nnabla_version/value.text = nn_crt_simple.rt_nnabla_version()
    $lbl_c_runtime_version/value.text = nn_crt_simple.rt_c_runtime_version()
    $lbl_nnb_version/value.text = str(nn_crt_simple.rt_nnb_version())
    $lbl_nnb_revision/value.text = nn_crt_simple.rt_nnb_revision()

    list_ui_edit = [
        $simple/edit_0, $simple/edit_1,
        $simple/edit_2, $simple/edit_3,
        $simple/edit_4, $simple/edit_5,
        $simple/edit_6, $simple/edit_7,
        $simple/edit_8, $simple/edit_9
    ]

    list_ui_label = [
        $simple/edit_0/label, $simple/edit_1/label,
        $simple/edit_2/label, $simple/edit_3/label,
        $simple/edit_4/label, $simple/edit_5/label,
        $simple/edit_6/label, $simple/edit_7/label,
        $simple/edit_8/label, $simple/edit_9/label
    ]

    list_ui_tex = [
        $mnist/tex0, $mnist/tex1, $mnist/tex2, $mnist/tex3
    ]


func _process(_delta):
    pass


func _on_btn_setup_pressed():

    for i in range(10):
        var f_value = randf()
        list_ui_edit[i].text = "%.2f" % f_value
        list_ui_label[i].text = ""

    ary_mnist_param.clear()
    for ui in list_ui_tex:
        var idx = randi_range(0, 255)
        ary_mnist_param.append(set_texture(ui, idx))

    $mnist/tex0/label.text = ""
    $mnist/tex1/label.text = ""
    $mnist/tex2/label.text = ""
    $mnist/tex3/label.text = ""


func _on_btn_forward_pressed():

    var output

    # simple
    ary_simple_param.clear()
    for i in range(10):
        ary_simple_param.append(float(list_ui_edit[i].text))
    
    nn_crt_simple.rt_input_buffer(0, ary_simple_param)

    var err = nn_crt_simple.rt_forward()
    assert(err == GDNNablaCRuntime.RT_RET_NOERROR)
    
    output = nn_crt_simple.rt_output_buffer(0)

    var v = -1
    var idx = 0
    for i in range(output.size()):
        list_ui_label[i].text = ""
        if output[i] > v:
            v = output[i]
            idx = i

    if v > -1 and v < 10:
        list_ui_label[idx].text = "o"

    # mnist
    var ary_result: Array = []

    for ary_param in ary_mnist_param:
        nn_crt_mnist.rt_input_buffer(0, ary_param)
        nn_crt_mnist.rt_forward()
        output = nn_crt_mnist.rt_output_buffer(0)
        ary_result.append(get_number(output))

    $mnist/tex0/label.text = "%d" % ary_result[0]
    $mnist/tex1/label.text = "%d" % ary_result[1]
    $mnist/tex2/label.text = "%d" % ary_result[2]
    $mnist/tex3/label.text = "%d" % ary_result[3]
