Initial Category:

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


Category-1: Data Loading

Category-1.1: Data loading

Candidate code elements:
model.load_state_dict(torch.load(data.load_model_dir)), data.load(data.dset_dir), pickle.load(fp), json.load(config_file), self.load_model_dir = None, ibrosa.load(filename, sr=sample_rate, mono=True), with open(args.wavenet_params, 'r') as f, load(saver, sess, restore_from), load_generic_audio(self.audio_dir, self.sample_rate), np.load(path), data_utils.DataLoader(), torch.load(checkpoint_path), load_audio(args.input_path), self.download_data(), download_an4(), wget.download(lm_path), latest_blob.download_to_filename(self.local_save_file), blob.upload_from_filename(self.local_save_file), AudioDataLoader(), if cfg.checkpointing.load_auto_checkpoint, visdom_logger.load_previous_values(state.epoch, state.results), TrainingState.load_state(), load_randomly_augmented_audio(audio_path, self.sample_rate), tf.saved_model.loader.load(), load_pretrained = hparams.load_pretrained, train_config.load_all_detection_checkpoint_vars, tf.resource_loader.get_data_files_path(), load_fine_tune_checkpoint(), _load_labelmap(), coco_gt.loadRes(predictions=coco_predictions), dataloader_utils.process_source_id(), mnist.download_and_prepare(), yaml.load(f), hub.load(params.hub_module_url), data_loader_factory.get_data_loader(dataloader_params).load(input_context), download_and_extract(), download_from_url(path, url), get_raw_files(FLAGS.data_dir, _TEST_DATA_SOURCES), _load_vocab_file(vocab_file, reserved_tokens), self.sp_model.Load(sp_model_file), TaggingDataLoader(data_config).load(), pd.read_csv(attributes_file), with h5py.File(hdf5_file, "a") as hf, hf.create_dataset("data_VGG_%s" % str(i),, data=vgg19_feat[i]), with open(d_act_path, 'r') as f, dset.MNIST(root=".", download=True), data_utils.load_celebA(img_dim, image_data_format), load_image_dataset(dset, img_dim, image_data_format), mnist_loader = get_loader(config), tl.files.load_file_list(), data = np.frombuffer(f.read(), np.uint8, offset=8), words = f.read(), _download(filename, working_directory, url_source)
tf.io.read_file(filename)
tf.data.Dataset.from_tensor_slices(filenames)
open(input_file,'r', encoding="utf8").readlines()

Category-1.2: Model loading

Candidate code elements:
load_model(cls, path), load_model_package(cls, package), load_decoder(labels, cfg: LMConfig), load_previous_values(self, start_epoch, results_state), DeepSpeech.load_model_package(package), model.load_weights(latest_checkpoint), copy.deepcopy(clean_model.get_weights()), tf.keras.models.load_model(model_weights_path), base_model = VGG19(weights='imagenet', include_top=False), load_pretrained(model, num_classes, settings), model.load_state_dict(torch.load(data.load_model_dir)), tag_seq = model(), model = SeqLabel(data), args1, auxs1 = load_checkpoint(prefix1, epoch1), load_param(prefix, begin_epoch, convert=True)
Category-2: Prediction Model
Category-2.1: Model Input/ Model features
Candidate code elements:
batch_size = data.HP_batch_size, features = [np.asarray(sent[1]) for sent in input_batch_list], input_shapes = []
Category-2.2: Input Labeling
labels = [sent[3] for sent in input_batch_list]

Category-2.2: Model Parameter Tuning

Candidate code elements:
optimizer = optim.SGD(), config.TEST.test_epoch = 0, model.compile(loss='mse', optimizer='sgd'), model.fit(train_data, train_data, epochs=4, verbose=0)

Category-2.3: Model Architecture

Candidate code elements:
model_builder.SSD_FEATURE_EXTRACTOR_CLASS_MAP, configs = self.get_model_configs_from_proto(), model = tf.keras.models.Sequential()

Category-2.4: Model Training

Candidate code elements:
model.train(), config.TRAIN.model_prefix, self.create_train_model(configs['model'], configs['lstm_model']), model.fit_generator()
Category-2.5: Model Prediction/ Model Queries
Candidate code elements:
prediction_dict = model.predict(preprocessed_video, true_image_shapes), pred_results += pred_label, predicted_label_id = classifier.predict_class(audio_path)

Category-2.6: Model Output

Candidate code elements:
model.summary(), data.show_data_summary(), output_data = interpreter.get_tensor(output_details[0]['index']), model._feature_extractor
Category-2.7: Model Feedback
Candidate code elements:
pred_scores = evaluate(data, model, name, data.nbest), model.eval(), torch.save(model.state_dict(), model_name), es = self.score(), confusion_matrix(y_test, y_predict, labels = [x for x in range(n_classes)]), f1_score(y_test, y_predict, average = None, labels = [x for x in range(n_classes)], f1_score(y_test, y_predict, average='macro'), accuracy_score(y_test, y_predict), total_loss = (reduced_loss + l2_regularization_strength * l2_loss), classification_loss(
        truth=truth, predicted=predicted, weights=weights, is_one_hot=True)

Category-3: Learning Algorithm

Candidate code elements:
LogisticRegression(C=c, penalty=penalty, max_iter=max_iter, random_state=seed), PCA(n_components=2), nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

Category-4: Data Pipelines

Candidate code elements:
argparse.ArgumentParser(description='Input pipeline'), pipeline_config = pipeline_pb2.TrainEvalPipelineConfig(), get_configs_from_pipeline_file(FLAGS.pipeline_config_path), configs['model'] = pipeline_config.model, data.Field(sequential=True, preprocessing=data.Pipeline(), input_pipeline_context=None, configs['model'] = pipeline_config.model

Category-5: Reinforcement learning

Category-5.1: Environment

Candidate code elements:
os.getenv('TACOTRON_BUILD_VERSION'), os.environ.get("LOCAL_RANK"), raise EnvironmentError(), env = Environment(loader=FileSystemLoader("./")), home = os.environ['CUDA_PATH'], logger.info("\n" + collect_env_info()), env_tests.get_config_root_path(), module.setup_environment(), env_str = get_pretty_env_info(), get_reward_fn(env_name), return self.base_env.action_space, EnvWithGoal(create_maze_env.create_maze_env(env_name).gym,env_name), obs = env.reset(), run_environment(FLAGS.env, FLAGS.episode_length, FLAGS.num_episodes)
tf_env.current_obs(), tf_env = create_maze_env.TFPyEnvironment(environment), 'image': task_env.ModalityTypes.IMAGE, assert isinstance(env, task_env.TaskEnv), return self._env.observation(xytheta), g = self._env.graph, env = env_fn(), env.close(), env.observation_space.shape[0], env.action_space.n, self._env.graph

Category-5.2: Observation

Candidate code elements:
num_inputs = env.observation_space.shape[0], observation_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype), observation, reward, done, info = old_step(action), if len(observations) != len(observation_states):, observations, states, path = self._exploration()

Category-5.3: Action Selection

Candidate code elements:
num_actions = env.action_space.shape[0], observation, reward, done, info = self.env.step(action), action_index = self._env.action(), selected_actions = np.argmax(actions, axis=-1), 'a': env.actions.index('left'),   if action == 'stop':, obs, reward, done, info = self._task.reward(obs, done, info)

