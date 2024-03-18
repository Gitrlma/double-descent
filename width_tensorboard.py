from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
import os

start_path = 'mcnn/epoch_wise/'
cut = 'best'
width_start = f'mcnn/width_wise/{cut}/'
writer_40k_c = SummaryWriter(log_dir = width_start + '40k_coloured')
writer_40k_g = SummaryWriter(log_dir = width_start + '40k_grayscale')
writer_50k_c = SummaryWriter(log_dir = width_start + '50k_coloured')
writer_50k_g = SummaryWriter(log_dir = width_start + '50k_grayscale')
w4c_train = {}
w4g_train = {}
w5c_train = {}
w5g_train = {}
w4c_test = {}
w4g_test = {}
w5c_test = {}
w5g_test = {}
if cut == 'last':
    best_train_idx = -1
    best_test_idx = -1
for width in os.listdir(start_path):
    current_width = int(width.split('_')[1])
    for setting in os.listdir(start_path + width):
        for tb in os.listdir(start_path + width + '/' + setting):
            ea = event_accumulator.EventAccumulator(
                start_path + width + '/' + setting + '/' + tb,
                size_guidance={  # see below regarding this argument
                    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                    event_accumulator.IMAGES: 4,
                    event_accumulator.AUDIO: 4,
                    event_accumulator.SCALARS: 0,
                    event_accumulator.HISTOGRAMS: 1,
                })

            ea.Reload()
            train_loss = ea.Scalars('train_epoch_loss')
            test_loss = ea.Scalars('test_epoch_loss')

            if cut == 'best':
                best_train = 1000
                best_train_idx = 0
                best_test = 1000
                best_test_idx = 0
                for j in range(len(train_loss)):
                    if train_loss[j].value < best_train:
                        best_train_idx = j
                        best_train = train_loss[j].value
                    if test_loss[j].value < best_test:
                        best_test_idx = j
                        best_test = test_loss[j].value
            if setting == '40k_coloured':
                w4c_train[current_width] = train_loss[best_train_idx].value
                w4c_test[current_width] = test_loss[best_test_idx].value
            elif setting == '40k_grayscale':
                w4g_train[current_width] = train_loss[best_train_idx].value
                w4g_test[current_width] = test_loss[best_test_idx].value
            elif setting == '50k_coloured':
                w5c_train[current_width] = train_loss[best_train_idx].value
                w5c_test[current_width] = test_loss[best_test_idx].value
            elif setting == '50k_grayscale':
                w5g_train[current_width] = train_loss[best_train_idx].value
                w5g_test[current_width] = test_loss[best_test_idx].value


w4c_train = dict(sorted(w4c_train.items()))
w4g_train = dict(sorted(w4g_train.items()))
w5c_train = dict(sorted(w5c_train.items()))
w5g_train = dict(sorted(w5g_train.items()))
w4c_test = dict(sorted(w4c_test.items()))
w4g_test = dict(sorted(w4g_test.items()))
w5c_test = dict(sorted(w5c_test.items()))
w5g_test = dict(sorted(w5g_test.items()))

for w in w4c_train:
    # 40k_coloured
    writer_40k_c.add_scalar("train_loss", w4c_train[w], w)
    writer_40k_c.add_scalar("test_loss", w4c_test[w], w)
    # 40k_grayscale
    writer_40k_g.add_scalar("train_loss", w4g_train[w], w)
    writer_40k_g.add_scalar("test_loss", w4g_test[w], w)
    # 50k_coloured
    writer_50k_c.add_scalar("train_loss", w5c_train[w], w)
    writer_50k_c.add_scalar("test_loss", w5c_test[w], w)
    # 50k_grayscale
    writer_50k_g.add_scalar("train_loss", w5g_train[w], w)
    writer_50k_g.add_scalar("test_loss", w5g_test[w], w)


writer_40k_c.close()
writer_40k_g.close()
writer_50k_c.close()
writer_50k_g.close()
