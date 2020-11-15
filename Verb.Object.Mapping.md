# Map Code Snippets to Verb Object Pairs 

## Akond Rahman 

### Needed to write methodology 

Use the following table to find what are verb object pairs 

### Mapping 
| Code Snippet | Verb | Object | Event | Final Category |
|--------------|------|--------------------|---------------------|-------------------|
| torch.load() | load | data.load_model_dir|   Load training data from a directory| Load training data|       
| data.load()  | load | data.dset_dir      |   Load training data from a directory| Load training data|       
| pickle.load()| load | fp                 |   Load training data from a file     | Load training data|       
| json.load()  | load | config_file        |   Load training data from a file     | Load training data|       
| np.load()    | load | path               |   Load training data| Load training data|       
| pickle.load()| load | fp                 |   Load training data| Load training data|       
| blob.upload_from_filename()| upload_from_filename | self.local_save_file | Upload training data from local file| Load training data|   
| yaml.load()  | load | f                  |   Load training data| Load training data|       
| hub.load()   | load | params.hub_module_url|   Load training data from remote source | Load training data|       
| data_loader_factory.get_data_loader().load()| load | dataloader_params, input_context |   Load training data| Load training data|       
| tf.io.read_file()   | read_file | filename|   Load training data from local file | Load training data| 
| tf.data.Dataset.from_tensor_slices() | from_tensor_slices() | filenames|   Load training data from local file | Load training data| 
| TaggingDataLoader().load()   | load | data_config|   Load training data from local file | Load training data| 
| pd.read_csv()   | read_csv | attributes_file|   Load training data from local CSV file | Load training data| 
| ibrosa.load()   | load() | filename|   Load training data from local CSV file | Load training data| 
| dset.MNIST()    | MNIST() | .|   Load training data from local directory | Load training data| 
| tarfile.open()  | open() | target_file|   Load training data from local directory | Load training data| 
| audio.load_wav()| load_wav() | wav_path|   Load training data from local directory | Load training data| 
| Image.open() | open() | args.demo_image|   Load training data from local directory | Load training data| 
| agent.replay_buffer.load()  | load() | self.rbuf_filename|   Load training data from local directory | Load training data| 
| h5py.File()  | File() | hdf5_file|   Load training data of H5 binary type from local directory | Load training data| 


- latest_blob.download_to_filename(self.local_save_file)
- coco_gt.loadRes(predictions=coco_predictions)
- self.sp_model.Load(sp_model_file)


### Need to exclude from FAME-ML 



- visdom_logger.load_previous_values(state.epoch, state.results)
- tl.files.load_file_list() 
- data_utils.load_celebA(img_dim, image_data_format)
