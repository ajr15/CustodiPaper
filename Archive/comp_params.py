rnn = {
        'tokenization_ds': [
						[{'tokenization_method': ['one_hot']}],
						[{'tokenization_method': ['aug_custodi'], 'alpha': [0.01, 0.05]}]
						],
    		'model_ds': [{
    					'model': ['RNN'], 
    					'lr': [0.01, 0.1], 
    					'dropout_rate': [0, 0.1]
    					}]
		}
		
nn_small = {
			'tokenization_ds': [
							[{'tokenization_method': ['word']}],
							[{'tokenization_method': ['custodi'], 'degree': [1, 2], 'alpha': [0.01, 0.05]}]
							],
			'model_ds': [{
						'model': ['NN'], 
						'lr': [0.01, 0.1], 
						'dropout_rate': [0, 0.1]
						}]
			}
		
nn_medium = {
			'tokenization_ds': [
							[{'tokenization_method': ['ECFP4']}],
							[{'tokenization_method': ['one_hot']}],
							[{'tokenization_method': ['aug_custodi'], 'alpha': [0.01, 0.05]}],
							[{'tokenization_method': ['cm']}],
							[{'tokenization_method': ['random']}]
							],
			'model_ds': [{
						'model': ['NN'], 
						'lr': [0.01, 0.1], 
						'dropout_rate': [0, 0.1]
						}]
			}
		
krr_small = {
			'tokenization_ds': [
							[{'tokenization_method': ['word']}],
							[{'tokenization_method': ['custodi'], 'degree': [1, 2], 'alpha': [0.01, 0.05]}]
							],
			'model_ds': [{
						'model': ['KRR'], 
						'model_alpha': [0.01, 0.1], 
						'kernel': ['rbf']
						}]
			}
		
krr_large = {
			'tokenization_ds': [
							[{'tokenization_method': ['word']}],
							[{'tokenization_method': ['custodi'], 'degree': [1, 2], 'alpha': [0.01, 0.05]}],
							[{'tokenization_method': ['ECFP4']}],
							[{'tokenization_method': ['one_hot']}],
							[{'tokenization_method': ['aug_custodi'], 'alpha': [0.01, 0.05]}],
							[{'tokenization_method': ['cm']}],
							[{'tokenization_method': ['random']}]
							],
			'model_ds': [{
						'model': ['KRR'], 
						'model_alpha': [0.01, 0.1], 
						'kernel': ['rbf']
						}]
			}

custodi = {
    			'tokenization_ds': [[{'tokenization_method': ['None']}]],
    			'model_ds': [{
    						'model': ['custodi'], 
    						'model_alpha': [0, 0.01, 0.1], 
 						   'model_degree': [1, 2]
    						}]
    			}