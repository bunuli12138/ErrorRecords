## 错误详情
### 出错代码
```python
'''
tmp_x: X_train
preproc_plaintext_sentences: y_train
'''
simple_rnn_model.fit(tmp_x, preproc_plaintext_sentences, 
                      batch_size=32, epochs=5, validation_split=0.2)
```

### 报错信息（较长）
```python
InvalidArgumentError                      Traceback (most recent call last)
<ipython-input-16-d3b38dac11b9> in <module>()
----> 1 simple_rnn_model.fit(tmp_x, preproc_plaintext_sentences, batch_size=32, epochs=5, validation_split=0.2)

D:\setup_space_all\anaconda\lib\site-packages\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, **kwargs)
   1037                                         initial_epoch=initial_epoch,
   1038                                         steps_per_epoch=steps_per_epoch,
-> 1039                                         validation_steps=validation_steps)
   1040 
   1041     def evaluate(self, x=None, y=None,

D:\setup_space_all\anaconda\lib\site-packages\keras\engine\training_arrays.py in fit_loop(model, f, ins, out_labels, batch_size, epochs, verbose, callbacks, val_f, val_ins, shuffle, callback_metrics, initial_epoch, steps_per_epoch, validation_steps)
    197                     ins_batch[i] = ins_batch[i].toarray()
    198 
--> 199                 outs = f(ins_batch)
    200                 outs = to_list(outs)
    201                 for l, o in zip(out_labels, outs):

D:\setup_space_all\anaconda\lib\site-packages\keras\backend\tensorflow_backend.py in __call__(self, inputs)
   2713                 return self._legacy_call(inputs)
   2714 
-> 2715             return self._call(inputs)
   2716         else:
   2717             if py_any(is_tensor(x) for x in inputs):

D:\setup_space_all\anaconda\lib\site-packages\keras\backend\tensorflow_backend.py in _call(self, inputs)
   2673             fetched = self._callable_fn(*array_vals, run_metadata=self.run_metadata)
   2674         else:
-> 2675             fetched = self._callable_fn(*array_vals)
   2676         return fetched[:len(self.outputs)]
   2677 

D:\setup_space_all\anaconda\lib\site-packages\tensorflow\python\client\session.py in __call__(self, *args, **kwargs)
   1437           ret = tf_session.TF_SessionRunCallable(
   1438               self._session._session, self._handle, args, status,
-> 1439               run_metadata_ptr)
   1440         if run_metadata:
   1441           proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)

D:\setup_space_all\anaconda\lib\site-packages\tensorflow\python\framework\errors_impl.py in __exit__(self, type_arg, value_arg, traceback_arg)
    526             None, None,
    527             compat.as_text(c_api.TF_Message(self.status.status)),
--> 528             c_api.TF_GetCode(self.status.status))
    529     # Delete the underlying status object from memory otherwise it stays alive
    530     # as there is a reference to status from this from the traceback due to

InvalidArgumentError: Incompatible shapes: [3232] vs. [32,101]
	 [[{{node metrics/acc/Equal}}]]
```

### RNNs建模代码
```python
'''
input_shape:            value: (10001, 101, 1)
output_sequence_length: value: 101 
code_vocab_size:        value: 32 
plaintext_vocab_size:   value: 32
'''
def simple_model(input_shape, output_sequence_length, code_vocab_size, plaintext_vocab_size):

    # TODO: Build the layers
    learning_rate = 1e-3

    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences=True)(input_seq)
    logits = TimeDistributed(Dense(plaintext_vocab_size))(rnn)

    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model
```

### 环境版本
` * 运行失败指发生以上错误，运行成功指无以上错误。`

| 模块 | 运行失败电脑版本 | 运行成功电脑版本 | 
| --- | --- | --- | 
| Keras | 2.2.4 | 2.0.9 | 
| Tensorflow | 1.13.1 | 1.3.0 |

