
def get_stuff_from_mode(mode):
    assert mode in list("DLC")
    if mode == "L":
        from models import LinearSmoosh
        from layers import smoosh_linear
        from losses import is_sim_linear_loss
        kl = LinearSmoosh(model = smoosh_linear())
        loss = is_sim_linear_loss()
        saved_model_dir = "saved_models/smoosh_linear"
    elif mode == "C":
        from models import CrossentropySmoosh
        from layers import smoosh_classification
        from losses import is_sim_categorical_loss
        kl = CrossentropySmoosh( model = smoosh_classification())
        loss = is_sim_categorical_loss()
        saved_model_dir = "saved_models/smoosh_classifier"
    elif mode == "D":
        from models import KerasLinear
        from layers import default_n_linear
        kl = KerasLinear(model = default_n_linear())
        loss = "mse"
        saved_model_dir = "saved_models/default"
    else:
        assert False, "you fuckhead"

    kl.compile(loss=loss)
    return kl, saved_model_dir

