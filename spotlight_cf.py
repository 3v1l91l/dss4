import numpy as np
from spotlight.cross_validation import user_based_train_test_split
from spotlight.evaluation import sequence_mrr_score
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.datasets.synthetic import generate_sequential

dataset = generate_sequential(num_users=100,
                              num_items=1000,
                              num_interactions=10000,
                              concentration_parameter=0.01,
                              order=3)
train, test = user_based_train_test_split(dataset)

train = train.to_sequence()
test = test.to_sequence()

model = ImplicitSequenceModel(n_iter=3,
                              representation='cnn',
                              loss='bpr')
model.fit(train)

mrr = sequence_mrr_score(model, test)
