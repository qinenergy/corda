from model import deeplabv2_synthia, deeplabv2_gta, deeplabv2_multitask, deeplabv2_noatt


def get_model(model_name):
    if model_name == "deeplabv2_synthia":
        return deeplabv2_synthia.Res_Deeplab
    elif model_name == "deeplabv2_gta":
        return deeplabv2_gta.Res_Deeplab      
    elif model_name == "deeplabv2_multitask":
        return deeplabv2_multitask.Res_Deeplab     
    elif model_name == "deeplabv2_noatt":
        return deeplabv2_noatt.Res_Deeplab
    else:
        raise NotImplementedError(model_name)
