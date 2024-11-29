from datasets import load_dataset, concatenate_datasets
import numpy as np
import os
from tqdm import tqdm

num_proc = 6 # should be ~ 1/2 num_cores in your machine
mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
if __name__ == '__main__':
    # Load tiny-imagenet dataset
    ds = load_dataset('zh-plus/tiny-imagenet', num_proc=num_proc)
    # concatenate valid split into train split. remove this line if you want to keep the original split
    ds['train'] = concatenate_datasets([ds['train'], ds['valid']])
    
    def filter_grey_scale(example):
        return len(np.array(example['image']).shape) == 3
    
    def normalize(example):
        img = np.array(example['image']).astype(np.float32).transpose(2, 0, 1)
        # Normalize
        img = (img / 255.0 - mean) / std
        return {'image': img, 'label': example['label']}
    
    filtered_ds = ds.filter(filter_grey_scale, desc='Filtering grey scale images', num_proc=num_proc)
    # Apply normalization
    processed_ds = filtered_ds.map(
        normalize,
        desc='Normalizing images',
        num_proc=num_proc,
    )
    # shuffle the dataset
    processed_ds = processed_ds.shuffle(seed=42)
    

    # Save the processed dataset into binary files
    for split, dset in processed_ds.items():
        print(len(dset))
        arr_shape = (len(dset), 3, 64, 64)
        image_filename = os.path.join(os.path.dirname(__file__), f'{split}_img.bin')
        label_filename = os.path.join(os.path.dirname(__file__), f'{split}_label.bin')
        dtype = np.float32
        arr_img = np.memmap(image_filename, dtype=dtype, mode='w+', shape=arr_shape)
        arr_label = np.memmap(label_filename, dtype=np.int32, mode='w+', shape=(arr_shape[0],))
        total_batches = 100
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Saving dataset'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_img_batch = np.stack(batch['image'])
            arr_label_batch = np.array(batch['label'])
            arr_img[idx:idx+len(arr_img_batch)] = arr_img_batch
            arr_label[idx:idx+len(arr_label_batch)] = arr_label_batch
            idx += len(arr_img_batch)
        arr_img.flush()
        arr_label.flush()
            
    
    