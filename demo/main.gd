extends Control


var input := PackedFloat32Array()
var list_edit := []
var list_label := []

var nn_crt: GDNNablaCRuntime;
var nn_crt_mnist: GDNNablaCRuntime;
var ary_mnist_param: Array[PackedFloat32Array]


func set_texture(tex_rect: TextureRect, idx: int) -> PackedFloat32Array:

    var img = Image.load_from_file("res://res/t10k/img_%04d.png" % idx)
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


func init_mnist():

    var rf = FileAccess.open("res://nn_model/model_mnist.nnb", FileAccess.READ)
    var rf_size = rf.get_length()
    var net_nnb = rf.get_buffer(rf_size)
    rf.close()
    
    nn_crt_mnist = GDNNablaCRuntime.new()

    var err = nn_crt_mnist.rt_allocate_context()
    assert(err == GDNNablaCRuntime.NOERROR)

    nn_crt_mnist.rt_initialize_context(net_nnb)


func _ready():

    init_mnist()

    var rf = FileAccess.open("res://nn_model/model.nnb", FileAccess.READ)
    var rf_size = rf.get_length()
    var nn_crt_nnb = rf.get_buffer(rf_size)
    rf.close()

    list_edit = [
        $edit_0, $edit_1, $edit_2, $edit_3, $edit_4,
        $edit_5, $edit_6, $edit_7, $edit_8, $edit_9
    ]

    list_label = [
        $edit_0/label, $edit_1/label, $edit_2/label, $edit_3/label, $edit_4/label,
        $edit_5/label, $edit_6/label, $edit_7/label, $edit_8/label, $edit_9/label
    ]

    nn_crt = GDNNablaCRuntime.new()

    var err = nn_crt.rt_allocate_context()
    assert(err == GDNNablaCRuntime.NOERROR)

    nn_crt.rt_initialize_context(nn_crt_nnb)

    var dict_i = nn_crt.rt_input_variable(0)
    print(dict_i)

    var dict_o = nn_crt.rt_output_variable(0)
    print(dict_o)

    for i in range(nn_crt.rt_input_dimension(0)):
        print("idx:{0} shape:{1}".format([i, nn_crt.rt_input_shape(0, i)]))

    for i in range(nn_crt.rt_output_dimension(0)):
        print("idx:{0} shape:{1}".format([i, nn_crt.rt_output_shape(0, i)]))


func _process(_delta):
    pass


func _on_btn_setup_pressed():

    input.resize(10)

    for i in range(10):
        var f_value = randf()
        input.set(i, f_value)
        list_edit[i].text = "%.2f" % f_value

    if true:
        ary_mnist_param.clear()
        for ui in [$mnist/tex0, $mnist/tex1, $mnist/tex2, $mnist/tex3]:
            var idx = randi_range(0, 255)
            ary_mnist_param.append(set_texture(ui, idx))


func _on_btn_forward_pressed():

    var ary_result: Array = []

    if true:
        for ary_param in ary_mnist_param:
            nn_crt_mnist.rt_input_buffer(0, ary_param)
            nn_crt_mnist.rt_forward()
            var output = nn_crt_mnist.rt_output_buffer(0)
            ary_result.append(get_number(output))

        $mnist/tex0/label.text = "%d" % ary_result[0]
        $mnist/tex1/label.text = "%d" % ary_result[1]
        $mnist/tex2/label.text = "%d" % ary_result[2]
        $mnist/tex3/label.text = "%d" % ary_result[3]

    nn_crt.rt_input_buffer(0, input)
    
    var err = nn_crt.rt_forward()
    assert(err == GDNNablaCRuntime.NOERROR)
    
    var output = nn_crt.rt_output_buffer(0)

    var v = -1
    var idx = 0
    for i in range(output.size()):
        list_label[i].text = ""
        if output[i] > v:
            v = output[i]
            idx = i

    if v > -1 and v < 10:
        list_label[idx].text = "o"
