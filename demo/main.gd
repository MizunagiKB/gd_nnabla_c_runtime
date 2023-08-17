extends Control


var input := PackedFloat32Array()
var net_nnb

var list_edit := []
var list_label := []


func _ready():

    var rf = FileAccess.open("res://nn_model/model.nnb", FileAccess.READ)
    var rf_size = rf.get_length()
    net_nnb = rf.get_buffer(rf_size)
    rf.close()

    list_edit = [
        $edit_0, $edit_1, $edit_2, $edit_3, $edit_4,
        $edit_5, $edit_6, $edit_7, $edit_8, $edit_9
    ]

    list_label = [
        $label_0, $label_1, $label_2, $label_3, $label_4,
        $label_5, $label_6, $label_7, $label_8, $label_9
    ]


func _process(_delta):
    pass


func _on_btn_setup_pressed():

    input.resize(10)

    for i in range(10):
        var f_value = randf()
        input.set(i, f_value)
        list_edit[i].text = "%.2f" % f_value


func _on_btn_forward_pressed():

    var nn_crt = GDNNablaCRuntime.new()

    var err

    err = nn_crt.rt_allocate_context()
    assert(err == GDNNablaCRuntime.NOERROR)

    nn_crt.rt_initialize_context(net_nnb)

    nn_crt.rt_input_buffer(0, input)
    
    err = nn_crt.rt_forward()
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

    nn_crt.rt_free_context()
