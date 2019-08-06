import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
import h5py
import numpy as np
from skimage import util

print("Matplot backend",  matplotlib.get_backend())

CLOCK = 12.5
DRIFT_VEL = 5.2
point_cutoff = 20


def clean_filter(xyz):
    filtered_1 = xyz[xyz[:, 6] < 100]
    filtered_2 = filtered_1[(filtered_1[:, 2]*DRIFT_VEL/CLOCK) < 1250.]
    filtered_3 = filtered_2[filtered_2[:, 5] > 0.001]
    return filtered_3

def real_filter(xyz):
    return xyz


def make_images(projection, image_size=int(128), clean=True):
    if clean:
        filtering = clean_filter
        dirname = "clean"
    else:
        filtering = real_filter
        dirname = "real"

    for labeled in [True, False]:
        events_all = []
        max_charge_all = []
        if labeled:
            runs = ["0210", "0130"]
        else:
            runs = ["0210", "0130", "0150", "0190"]
        for run in runs:
            print("Starting... ")
            events = []
            file = h5py.File("../data/clean/runs/clean_run_{}.h5".format(run))
            dataset = file["/clean"]
            discarded_events = ["run_{}_label_{}".format(run, labeled)]

            if labeled:
                labels = pd.read_csv("../labels/run_{}_labels.csv".format(run))

                proton_indices = labels.loc[(
                    labels['label'] == 'p')]['evt_id'].values
                carbon_indices = labels.loc[(
                    labels['label'] == 'c')]['evt_id'].values
                junk_indices = labels.loc[(
                    labels['label'] == 'j')]['evt_id'].values

                for evt_id in carbon_indices:
                    event = dataset[str(evt_id)]
                    xyzs = filtering(np.array(event))

                    if len(xyzs) < point_cutoff:
                        discarded_events.append(evt_id)
                    else:
                        events.append([xyzs, 1])

                for evt_id in proton_indices:
                    event = dataset[str(evt_id)]
                    xyzs = filtering(np.array(event))

                    if len(xyzs) < point_cutoff:
                        discarded_events.append(evt_id)
                    else:
                        events.append([xyzs, 0])
                n_keyerr = 0 
                for evt_id in junk_indices:
                    try:
                        event = dataset[str(evt_id)]
                    except KeyError:
                        n_keyerr += 1
                        discarded_events.append(evt_id)
                        continue

                    xyzs = filtering(np.array(event))

                    if len(xyzs) < point_cutoff:
                        discarded_events.append(evt_id)
                    else:
                        events.append([xyzs, 2])

                print("n events disc. to keyErr", n_keyerr)

            else:
                for i in dataset:
                    event = dataset[str(i)]
                    xyzs = filtering(np.array(event))

                    if len(xyzs) < point_cutoff:
                        discarded_events.append(i)
                    else:
                        events.append([xyzs, 1])

            print("Discarding and scaling... ", run)
            print("N Discarded : ", len(discarded_events))

            """
            Discard events with low point number
            """
            discarded_events = np.array(discarded_events)

            np.save(
                "../data/"+dirname+"/discarded/discarded_events_{}_label_{}.npy".format(run, labeled),
                discarded_events)

            """
            log ( 1+x) of the charge
            """
            log_charge_events = Parallel(n_jobs=8)(
                delayed(np.log1p)(event[0][:, 3]) for event in events)

            for i in range(len(log_charge_events)):
                events[i][0][:, 3] = log_charge_events[i]

            """
            normalize charge
            """
            max_charge = np.array(
                list(map(lambda x: x[0][:, 3].max(), events))).max()
            max_charge_all.append(max_charge)
            events_all.append(np.array(events))

        for events, run in zip(events_all, runs):
            print("----------------")
            print("Saving Images : ", run,)
            print("Type: ", dirname)
            print("Labeled: ", labeled)
            print("N_evemts: ", events.shape,) 
            print("-----------------")
            max_charge = max(max_charge_all)
            normalized_charge_events = Parallel(n_jobs=8)(
                delayed(lambda x: x/max_charge)(event[0][:, 3]) for event in events)

            for i in range(len(normalized_charge_events)):
                events[i][0][:, 3] = normalized_charge_events[i]
            
            print(type(image_size))
            images = np.empty((len(events), image_size, image_size, 1), dtype=np.uint8)
            targets = np.empty(len(events), dtype=np.uint8)

            def make_image(event):
                e = event[0]
                t = event[1]

                if projection == 'zy':
                    x = e[:, 2].flatten()
                    z = e[:, 1].flatten()
                    c = e[:, 3].flatten()
                elif projection == 'xy':
                    x = e[:, 0].flatten()
                    z = e[:, 1].flatten()
                    c = e[:, 3].flatten()
                else:
                    raise ValueError("Invalid projection value.")

                fig = plt.figure(figsize=(1, 1), dpi=image_size)
                if projection == 'zy':
                    plt.xlim(0.0, 1250.0)
                elif projection == 'xy':
                    plt.xlim(-275.0, 275.0)
                plt.ylim((-275.0, 275.0))
                plt.axis('off')
                plt.scatter(x, z, s=0.6, c=c, cmap='Greys')
                fig.canvas.draw()
                data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
                data = np.delete(data, 3, axis=2)
                data = util.invert(data[:, :, 0])
                data = np.expand_dims(data, -1)
                plt.close()

                return data, t

            print("Making images...")

            image_target = Parallel(n_jobs=8)(delayed(make_image)(event)
                                            for event in events)

            for i, i_t in enumerate(image_target):
                image, target = i_t
                images[i] = image
                targets[i] = target

            print("Saving...")
            np.save("../data/"+dirname+"/images/run_{}_label_{}_size_{}.npy".format(
                        run,
                        labeled,
                        image_size
                        ), images/255)

            if labeled:
                np.save(
                        "../data/"+dirname+"/targets/run_{}_targets_size_{}.npy".format(run, image_size),
                        targets
                        )


if __name__ == "__main__":
    make_images("xy", clean=True)
