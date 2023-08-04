import os
import re
import glob
import argparse
from keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau,EarlyStopping
from keras.optimizers import SGD,Adam
from model.Datagenerater   import DataGenerator
from model.MapsNet import MapsNet
from model.CDNet import CDNet
from model.loss import dice_coef ,precision,recall,bce_dice_loss,F1,weighted_bce_dice_loss
from model.FC_Siam_differ import FC_Siam_differ
from model.FC_Siam_conc import FC_Siam_conc

from model.CDNet import CDNet
from model.fsc import get_FCSC_model
from model.FCN_PP import FCN_PP
from model.fsd import get_FCSD_model


def get_images(data_path):
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'tif']:
        files.extend(glob.glob(os.path.join(data_path, '**/*.{}'.format(ext)), recursive=True))
    return files


def get_latest_model(check_dir):
    ckpts = os.listdir(os.path.join(check_dir))
    max_epoch = 0
    latest_model = ''
    for ckpt in ckpts:
        if re.search('epoch(\d+)', ckpt) is None:
            continue
        epoch = int(re.search('epoch(\d+)', ckpt).group(1))
        if epoch > max_epoch:
            max_epoch = epoch
            latest_model = os.path.join(check_dir, ckpt)
    return latest_model, max_epoch


def parse_args():
    parser = argparse.ArgumentParser('unet segmentation')
    parser.add_argument('--data_path', type=str, default='../DSFIN2/train', help='data path')
    parser.add_argument('--train1', type=str, default='t1', help='train images1 folder')
    parser.add_argument('--train2', type=str, default='t2', help='train images2 folder')
    parser.add_argument('--label', type=str, default='label512', help='train labels folder')
    parser.add_argument('--data2_path', type=str, default='../DSFIN2/val', help='data2 path')
    parser.add_argument('--val1', type=str, default='t1', help='test images1 folder')
    parser.add_argument('--val2', type=str, default='t2', help='test images2 folder')
    parser.add_argument('--val_label', type=str, default='label512', help='test labels folder')
    parser.add_argument('--size', type=int, default=512, help='input size')
    # parser.add_argument('--ClassNum',type = int,default= 3,help = 'ClassNum')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='epochs')
    parser.add_argument('--val', default =True, help='val_start')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='whether to resume training')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    BS = args.batch_size
    EPOCHS = args.epoch
    SIZE = args.size
    train_image1_len = len(get_images(os.path.join(args.data_path, args.train1)))
    train_image2_len = len(get_images(os.path.join(args.data_path, args.train2)))
    train_label_len = len(get_images(os.path.join(args.data_path, args.label)))

    print('training1 images: {}'.format(train_image1_len))
    print('training2 images: {}'.format(train_image2_len))
    print('training labels: {}'.format(train_label_len))
    train_gen = DataGenerator(args.data_path,
                              args.train1,
                              args.train2,
                              args.label,
                              BS,
                              target_size=(SIZE, SIZE),
                              augment=True,
                              shuffle=True)

    if args.val:
        val_image1_len = len(get_images(os.path.join(args.data2_path, args.val1)))
        val_image2_len = len(get_images(os.path.join(args.data2_path, args.val2)))
        val_label_len = len(get_images(os.path.join(args.data2_path, args.val_label)))
        print('val1 images: {}'.format(val_image1_len))
        print('val2 images: {}'.format(val_image2_len))
        print('val labels: {}'.format(val_label_len))
        val_gen = DataGenerator(args.data2_path,
                                args.val1,
                                args.val2,
                                args.val_label,
                                BS,
                                target_size=(SIZE, SIZE),
                                augment=True,
                                shuffle=True)
        val_steps = val_image1_len // BS
    else:
        val_gen = None
        val_steps = None

    check_dir = 'checkpoints'
    log_path = 'logs'
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    check_path = os.path.join(check_dir, 'FCD_{epoch:03d}.h5')

    start_epoch = 0
    if args.resume:
        model_file, start_epoch = get_latest_model(check_dir)
        model = get_FCSD_model()
        model.load_weights(model_file)
        print('Resume training from {}'.format(model_file))
    else:
        # model = vgg_unet(n_classes=1, input_height=SIZE, input_width=SIZE)
        model = get_FCSD_model()

    model.compile(optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam'),
                  loss=weighted_bce_dice_loss,
                  metrics=['accuracy', recall, precision, F1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint(check_path, monitor='val_loss',verbose=1, save_best_only=True,
                                       save_weights_only=True, period=1)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                             patience=5, verbose=1)
    tfboard_cb = TensorBoard(log_dir=log_path, write_graph=True, write_images=True)
    model.fit(train_gen,
            steps_per_epoch=train_image1_len // BS,
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=val_steps,
            initial_epoch=start_epoch,
            callbacks=[model_checkpoint, tfboard_cb, early_stopping, reduce_lr_on_plateau])


