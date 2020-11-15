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
| np.frombuffer()  | frombuffer() | f.read|   Load training data from local directory | Load training data| 
| get_loader()  | get_loader() | config |   Load training data from local directory | Load training data| 
| load_randomly_augmented_audio()  | load_randomly_augmented_audio() | audio_path |   Load  audio data for training from local directory | Load training data| 
| open()  | open() | input_file |   Load  file for training from local directory | Load training data| 
| open()  | open() | args.wavenet_params |   Load wavenet file for training from local directory | Load training data| 
| load_generic_audio(self.audio_dir)  | load_generic_audio() | self.audio_dir |   Load audio file for training  | Load training data|
| load_audio(args.input_path)  | load_audio() | args.input_path |   Load audio file for training  | Load training data| 
| load_audio()  | load_audio() | dset |   Load image file for training  | Load training data| 
| _load_vocab_file()  | _load_vocab_file() | vocab_file |   Load vocabulary file file for training  | Load training data| 
| read_h5file(os.path.join()   | read_h5file() | train.h5 |  Load H5 binary file file for training  | Load training data| 


- latest_blob.download_to_filename(self.local_save_file)
- _download(filename, working_directory, url_source)
- download_from_url(path, url)
- coco_gt.loadRes(predictions=coco_predictions)
- self.sp_model.Load(sp_model_file)


### Need to exclude from FAME-ML 



- visdom_logger.load_previous_values(state.epoch, state.results)
- tl.files.load_file_list() 
- load(saver, sess, restore_from)
- data_utils.load_celebA(img_dim, image_data_format)
- get_raw_files(FLAGS.data_dir, _TEST_DATA_SOURCES)
- load_attribute_dataset(args.attr_file) 
- load_lua(args.input_t7) 
