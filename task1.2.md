# Initial Category:

1.	Data Loading
2.	Model Training
3.	Model Prediction
4.	Model Tuning (Model Parameters)
5.	Model Input
6.	Model output
7.	Model Features
8.	Model Queries
9.	Learning Algorithm
10.	 Environment
11.	 Data Pipelines
12.	 Model Feedback
13.	 Model Architecture
14.	Observation
15.	Action Selection


## Category-1: Data Loading

### Category-1.1: Data loading

Candidate code elements:
model.load_state_dict(torch.load(data.load_model_dir)), data.load(data.dset_dir), pickle.load(fp), json.load(config_file), self.load_model_dir = None, ibrosa.load(filename, sr=sample_rate, mono=True), with open(args.wavenet_params, 'r') as f, load(saver, sess, restore_from), load_generic_audio(self.audio_dir, self.sample_rate), np.load(path), data_utils.DataLoader(), torch.load(checkpoint_path), load_audio(args.input_path), self.download_data(), download_an4(), wget.download(lm_path), latest_blob.download_to_filename(self.local_save_file), blob.upload_from_filename(self.local_save_file), AudioDataLoader(), if cfg.checkpointing.load_auto_checkpoint, visdom_logger.load_previous_values(state.epoch, state.results), TrainingState.load_state(), load_randomly_augmented_audio(audio_path, self.sample_rate), tf.saved_model.loader.load(), load_pretrained = hparams.load_pretrained, train_config.load_all_detection_checkpoint_vars, tf.resource_loader.get_data_files_path(), load_fine_tune_checkpoint(), _load_labelmap(), coco_gt.loadRes(predictions=coco_predictions), dataloader_utils.process_source_id(), mnist.download_and_prepare(), yaml.load(f), hub.load(params.hub_module_url), data_loader_factory.get_data_loader(dataloader_params).load(input_context), download_and_extract(), download_from_url(path, url), get_raw_files(FLAGS.data_dir, _TEST_DATA_SOURCES), _load_vocab_file(vocab_file, reserved_tokens), self.sp_model.Load(sp_model_file), TaggingDataLoader(data_config).load(), pd.read_csv(attributes_file), with h5py.File(hdf5_file, "a") as hf, hf.create_dataset("data_VGG_%s" % str(i),, data=vgg19_feat[i]), with open(d_act_path, 'r') as f, dset.MNIST(root=".", download=True), data_utils.load_celebA(img_dim, image_data_format), load_image_dataset(dset, img_dim, image_data_format), mnist_loader = get_loader(config), tl.files.load_file_list(), data = np.frombuffer(f.read(), np.uint8, offset=8), words = f.read(), _download(filename, working_directory, url_source)
tf.io.read_file(filename)
tf.data.Dataset.from_tensor_slices(filenames)
open(input_file,'r', encoding="utf8").readlines()

### Category-1.2: Model loading

Candidate code elements:
load_model(cls, path), load_model_package(cls, package), load_decoder(labels, cfg: LMConfig), load_previous_values(self, start_epoch, results_state), DeepSpeech.load_model_package(package), model.load_weights(latest_checkpoint), copy.deepcopy(clean_model.get_weights()), tf.keras.models.load_model(model_weights_path), base_model = VGG19(weights='imagenet', include_top=False), load_pretrained(model, num_classes, settings), model.load_state_dict(torch.load(data.load_model_dir)), tag_seq = model(), model = SeqLabel(data), args1, auxs1 = load_checkpoint(prefix1, epoch1), load_param(prefix, begin_epoch, convert=True)

## Category-2: Prediction Model

### Category-2.1: Model Input/ Model features

Candidate code elements:
batch_size = data.HP_batch_size, features = [np.asarray(sent[1]) for sent in input_batch_list], input_shapes = []

### Category-2.2: Input Labeling

Candidate code elements:
labels = [sent[3] for sent in input_batch_list]

### Category-2.3: Model Parameter Tuning

Candidate code elements:
optimizer = optim.SGD(), config.TEST.test_epoch = 0, model.compile(loss='mse', optimizer='sgd'), model.fit(train_data, train_data, epochs=4, verbose=0)

### Category-2.4: Model Architecture

Candidate code elements:
model_builder.SSD_FEATURE_EXTRACTOR_CLASS_MAP, configs = self.get_model_configs_from_proto(), model = tf.keras.models.Sequential()

### Category-2.5: Model Training

Candidate code elements:
model.train(), config.TRAIN.model_prefix, self.create_train_model(configs['model'], configs['lstm_model']), model.fit_generator()
Category-2.5: Model Prediction/ Model Queries
Candidate code elements:
prediction_dict = model.predict(preprocessed_video, true_image_shapes), pred_results += pred_label, predicted_label_id = classifier.predict_class(audio_path)

### Category-2.6: Model Output

Candidate code elements:
model.summary(), data.show_data_summary(), output_data = interpreter.get_tensor(output_details[0]['index']), model._feature_extractor

### Category-2.7: Model Feedback

Candidate code elements:
pred_scores = evaluate(data, model, name, data.nbest), model.eval(), torch.save(model.state_dict(), model_name), es = self.score(), confusion_matrix(y_test, y_predict, labels = [x for x in range(n_classes)]), f1_score(y_test, y_predict, average = None, labels = [x for x in range(n_classes)], f1_score(y_test, y_predict, average='macro'), accuracy_score(y_test, y_predict), total_loss = (reduced_loss + l2_regularization_strength * l2_loss), classification_loss(
        truth=truth, predicted=predicted, weights=weights, is_one_hot=True)

## Category-3: Learning Algorithm

Candidate code elements:
LogisticRegression(C=c, penalty=penalty, max_iter=max_iter, random_state=seed), PCA(n_components=2), nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

## Category-4: Data Pipelines

Candidate code elements:
argparse.ArgumentParser(description='Input pipeline'), pipeline_config = pipeline_pb2.TrainEvalPipelineConfig(), get_configs_from_pipeline_file(FLAGS.pipeline_config_path), configs['model'] = pipeline_config.model, data.Field(sequential=True, preprocessing=data.Pipeline(), input_pipeline_context=None, configs['model'] = pipeline_config.model

## Category-5: Reinforcement learning

### Category-5.1: Environment

Candidate code elements:
os.getenv('TACOTRON_BUILD_VERSION'), os.environ.get("LOCAL_RANK"), raise EnvironmentError(), env = Environment(loader=FileSystemLoader("./")), home = os.environ['CUDA_PATH'], logger.info("\n" + collect_env_info()), env_tests.get_config_root_path(), module.setup_environment(), env_str = get_pretty_env_info(), get_reward_fn(env_name), return self.base_env.action_space, EnvWithGoal(create_maze_env.create_maze_env(env_name).gym,env_name), obs = env.reset(), run_environment(FLAGS.env, FLAGS.episode_length, FLAGS.num_episodes)
tf_env.current_obs(), tf_env = create_maze_env.TFPyEnvironment(environment), 'image': task_env.ModalityTypes.IMAGE, assert isinstance(env, task_env.TaskEnv), return self._env.observation(xytheta), g = self._env.graph, env = env_fn(), env.close(), env.observation_space.shape[0], env.action_space.n, self._env.graph

### Category-5.2: Observation

Candidate code elements:
num_inputs = env.observation_space.shape[0], observation_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype), observation, reward, done, info = old_step(action), if len(observations) != len(observation_states):, observations, states, path = self._exploration()

### Category-5.3: Action Selection

Candidate code elements:
num_actions = env.action_space.shape[0], observation, reward, done, info = self.env.step(action), action_index = self._env.action(), selected_actions = np.argmax(actions, axis=-1), 'a': env.actions.index('left'),   if action == 'stop':, obs, reward, done, info = self._task.reward(obs, done, info)







# After Discussion


## Category-1: Data Loading

### Category-1.1: Data loading

Candidate code elements:
> model.load_state_dict(torch.load(data.load_model_dir)), data.load(data.dset_dir), pickle.load(fp), json.load(config_file), self.load_model_dir = None, ibrosa.load(filename, sr=sample_rate, mono=True), with open(args.wavenet_params, 'r') as f, load(saver, sess, restore_from), load_generic_audio(self.audio_dir, self.sample_rate), np.load(path), data_utils.DataLoader(), torch.load(checkpoint_path), load_audio(args.input_path), self.download_data(), download_an4(), wget.download(lm_path), latest_blob.download_to_filename(self.local_save_file), blob.upload_from_filename(self.local_save_file), if cfg.checkpointing.load_auto_checkpoint, visdom_logger.load_previous_values(state.epoch, state.results), TrainingState.load_state(), load_randomly_augmented_audio(audio_path, self.sample_rate), tf.saved_model.loader.load(), load_pretrained = hparams.load_pretrained, train_config.load_all_detection_checkpoint_vars, tf.resource_loader.get_data_files_path(), load_fine_tune_checkpoint(), _load_labelmap(), coco_gt.loadRes(predictions=coco_predictions), dataloader_utils.process_source_id(), mnist.download_and_prepare(), yaml.load(f), hub.load(params.hub_module_url), data_loader_factory.get_data_loader(dataloader_params).load(input_context), download_and_extract(), download_from_url(path, url), get_raw_files(FLAGS.data_dir, _TEST_DATA_SOURCES), _load_vocab_file(vocab_file, reserved_tokens), self.sp_model.Load(sp_model_file), TaggingDataLoader(data_config).load(), pd.read_csv(attributes_file), with h5py.File(hdf5_file, "a") as hf, with open(d_act_path, 'r') as f, dset.MNIST(root=".", download=True), data_utils.load_celebA(img_dim, image_data_format), load_image_dataset(dset, img_dim, image_data_format), mnist_loader = get_loader(config), tl.files.load_file_list(), data = np.frombuffer(f.read(), np.uint8, offset=8), words = f.read(), _download(filename, working_directory, url_source), tf.io.read_file(filename), tf.data.Dataset.from_tensor_slices(filenames), open(input_file,'r', encoding="utf8").readlines(), 

> tarfile.open(target_file), get_dataset(), audio.load_wav(wav_path), with open("cmuarctic.data", "r") as f:, utils.pickle_load(filename), cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE), load_attribute_dataset(args.attr_file), datasets.ImageFolder(), torch.utils.data.DataLoader(), Image.open(args.demo_image), load_image(), testLoader = DataLoader(), read_h5file(os.path.join(os.getcwd(), 'train.h5')), load_lua(args.input_t7),  glob.iglob(), codecs.open(), scipy.io.loadmat(), load_gt_roidb(), agent.replay_buffer.load(self.rbuf_filename), Image.fromArray(), tf.gfile.GFile(), dataset.ReadDatasetFile(), tf.io.gfile.glob()

### Category-1.2: Model loading

Candidate code elements:
> load_model(cls, path), load_decoder(labels, cfg: LMConfig), load_previous_values(self, start_epoch, results_state), DeepSpeech.load_model_package(package), tf.keras.models.load_model(model_weights_path), load_pretrained(model, num_classes, settings), model.load_state_dict(torch.load(data.load_model_dir)), tag_seq = model(), model = SeqLabel(data), args1, auxs1 = load_checkpoint(prefix1, epoch1), load_param(prefix, begin_epoch, convert=True)

> model_dir_path = patch_path('models'), ref = CaffeFunction('VGG_ILSVRC_19_layers.caffemodel'),load_state_dict(), model_from_json(open()), network.load_net(), vgg.load_from_npy_file(), caffe_parser.read_caffemodel() , tf.train.Checkpoint(), get_network(args.network_name), tfhub.load(), scipy.misc.imresize()

## Category-2: Data  downloads 

Candidate code elements: 
> wget.download('http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz'), urllib.request.urlopen(request), model_zoo.load_url(url), prepare_url_image(url), urllib.urlretrieve(), agent.load(misc.download_model())

## Category-3: Prediction Model

### Category-3.1: Model Input/ Model features

Candidate code elements:
batch_size = data.HP_batch_size, features = [np.asarray(sent[1]) for sent in input_batch_list], input_shapes = []

### Category-3.2: Label manipulation 

Candidate code elements:
- variable names with `label(s)` 
> *train_data, train_label = read_h5file(os.path.join(os.getcwd(), 'train.h5'))*
> *val_data, val_label = read_h5file(os.path.join(os.getcwd(), 'val.h5'))*
> *label = np.array(hf.get('label'))* 
> *label = load_image(f).convert('P')* 
> *label = scipy.io.loadmat('{}/segmentation/img_{}.mat'.format(self.nyud_dir, idx))['segmentation'].astype(np.uint8)* 
> *label = os.path.basename(os.path.dirname(one_file))* 
> *raw_data,raw_label = load_data_and_labels(fenci_right_save_path,fenci_wrong_save_path)*
> *label = hfw.create_dataset("labels", data=df_attr[list_col_labels].values)*
> *labels = [sent[3] for sent in input_batch_list]*

### Category-3.3: Model Output

Candidate code elements:
model.summary(), data.show_data_summary(), output_data = interpreter.get_tensor(output_details[0]['index']), model._feature_extractor

### Category-3.4: Model Feedback

Candidate code elements:
pred_scores = evaluate(data, model, name, data.nbest), model.eval(), es = self.score(), confusion_matrix(y_test, y_predict, labels = [x for x in range(n_classes)]), f1_score(y_test, y_predict, average = None, labels = [x for x in range(n_classes)], f1_score(y_test, y_predict, average='macro'), accuracy_score(y_test, y_predict), total_loss = (reduced_loss + l2_regularization_strength * l2_loss), classification_loss(
        truth=truth, predicted=predicted, weights=weights, is_one_hot=True)

## Category-4: Data Pipelines

Candidate code elements:
argparse.ArgumentParser(description='Input pipeline'), pipeline_config = pipeline_pb2.TrainEvalPipelineConfig(), get_configs_from_pipeline_file(FLAGS.pipeline_config_path), configs['model'] = pipeline_config.model, data.Field(sequential=True, preprocessing=data.Pipeline(), input_pipeline_context=None, configs['model'] = pipeline_config.model

## Category-5: Reinforcement learning

### Category-5.1: Environment

Candidate code elements: we need to log environment dynamics: (i) state, (ii) reward, (iii) action, and (iv) policy = <observation, action, reward, done>
> *inner_next_obs, inner_reward, done = wrapped_env.step(action)*
> *obs, reward, done, tt = env.step(action)* 
> *state, reward, done, _ = env.step(action)*
> *obs, reward, done, info = env.step(action)* 
> *next_state, reward, done, _ = env.step(action)* 
> *batch_ob, reward, done, info = self.env.step(action)* 
> *state, reward, done, _ = env.step(action)*
> *o, r, done, info = self.env.step(action)* 
> *policy = torch.load(args.model_path)* 
> *env = gym.make(params.env_name)* and *num_inputs = env.observation_space.shape[0]* and *num_outputs = env.action_space.shape[0]*

### Category-5.2: State Observation

Candidate code elements: `obs` or `state` or `observations` obtained from in env.step(action)[0]
> *obs, reward, done, tt = env.step(action)* 

### Category-5.3: Action Selection

Candidate code elements: `action` used in env.step()
> *obs, reward, done, tt = env.step(action)* 


## Category-6: Classification decision of DNNs

Candidate code elements:
-  `torch.nn` imported and used to perform classifications ... relatively harder to detect as many scripts are libraries, they provide utils but do not actually apply the library for classification decision 
- `from keras.models import *` Reference: https://realpython.com/python-keras-text-classification/ 
> *import torch.nn* or *import torch* and *nn.ReLU()* and *cls = PointNetCls(k = 5)* *out, _, _ = cls(sim_data)*
> *model=cascaded_model(D_m)* and *G=model(D, label_images)* and *Generator=G.permute(0,2,3,1)* and *output=np.minimum(np.maximum(Generator,0.0), 255.0)*
> *model = Model(inputs=[inputs], outputs=[conv10])* and *model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])* and *from keras.models import Model*
> *model = Graph()* and *from keras.models import *graph.compile()* and *graph.fit()* and *graph.predict()* 
> *model_from_file.fit()* and *model_from_file.evaluate()* and *from keras.models import Sequential, model_from_json*
> *graph = VGG_16_graph()* and *model = Graph()* and *from keras.models import ** and *model.compile()* and *graph.predict()* 


### Category-7: Incomplete Logging 

Candidate code elements:
- variable names with `import logging` 
> *logging.getLogger().setLevel(logging.DEBUG)*
> *logger = logging.getLogger() #no mention of timestamp*
> *logging.basicConfig() # timestamp not specified using FORMAT string*
> *from symnet.logger import logger* and *logger.info('called with args\n{}'.format(pprint.pformat(vars(args))))* 
> *import tensorflow.compat.v1 as tf* and *tf.logging()* 
> *tf.compat.v1.logging.info("Successfully downloaded %s, size(bytes): %d" % (url, statinfo.st_size))* and *import tensorflow as tf*

A good example that should happen : 
> *FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'  logging.basicConfig(format=FORMAT)*
