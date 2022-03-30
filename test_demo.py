import os.path
import logging
from collections import OrderedDict
import torch

from utils import utils_logger
from utils import utils_image as util
from utils.utils_modelsummary import get_model_activation, get_model_flops
from models.mobilesr import MobileSR


def main():

    utils_logger.logger_info('NTIRE2022-EfficientSR', log_path='NTIRE2022-EfficientSR-Test.log')
    logger = logging.getLogger('NTIRE2022-EfficientSR')

    # --------------------------------
    # basic settings
    # --------------------------------
    # testsets = 'DIV2K'
    testsets = os.path.join(os.getcwd(), 'data')
    testset_L = 'DIV2K_test_LR'

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------
    # load model
    # --------------------------------
    model_path = os.path.join('model_zoo', 'mobilesr_x4.pth')
    model = MobileSR()
    model.load_state_dict(torch.load(model_path)['net'], strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))
    
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv2d = get_model_activation(model, input_dim)
    logger.info('{:>16s} : {:<.4f} [M]'.format('#Activations', activations/10**6))
    logger.info('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))

    flops = get_model_flops(model, input_dim, False)
    logger.info('{:>16s} : {:<.4f} [G]'.format('FLOPs', flops/10**9))

    # --------------------------------
    # read image
    # --------------------------------
    L_folder = os.path.join(testsets, testset_L)
    E_folder = os.path.join(testsets, testset_L+'_results')
    util.mkdir(E_folder)

    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    logger.info(L_folder)
    logger.info(E_folder)
    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for img in util.get_image_paths(L_folder):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        start.record()
        img_E = model(img_L)
         
        end.record()
        torch.cuda.synchronize()
        test_results['runtime'].append(start.elapsed_time(end))  # milliseconds


#        torch.cuda.synchronize()
#        start = time.time()
#        img_E = model(img_L)
#        torch.cuda.synchronize()
#        end = time.time()
#        test_results['runtime'].append(end-start)  # seconds

        # --------------------------------
        # (2) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)

        util.imsave(img_E, os.path.join(E_folder, img_name[:4]+ext))
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))

if __name__ == '__main__':
#     from fvcore.nn import flop_count_str, flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
#     a = torch.randn(1, 3, 256, 256)
#     model = MobileSR() #IMDN(in_nc=3, out_nc=3, nc=64, nb=8, upscale=4)
#     print(flop_count_table(FlopCountAnalysis(model, a), activations=ActivationCountAnalysis(model, a)))
#     print(model(a).shape)

    main()