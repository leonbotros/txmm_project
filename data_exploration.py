
import matplotlib.pyplot as plt
import json

# Loads the data
train_data = json.load(open("train-v1.1.json"))['data']
dev_data = json.load(open("dev-v1.1.json"))['data']



question_words = ['What', 'Who', 'How', 'When', 'Which', 'Where', 'Why', 'Whose', 'Whom']
questions = []
print('Subjects: %d' % len(train_data))

for subj in train_data:
  title = subj['title']
  paragraphs = subj['paragraphs']
  for p in paragraphs:
    context = p['context']
    qas = p['qas']
    for q in qas:
      start = q['answers'][0]['answer_start']
      end = start + len(q['answers'][0]['text'])
      print(start, end)
      # print(q['answers'][0])
      questions.append(q['question'])

print(len(questions))
count = {x: len([q for q in questions if q.startswith(x)]) for x in question_words}
something_else = len(questions) - sum(count.values())

keys_sorted = sorted(count, key=count.get, reverse=True)
values = [count[x] for x in keys_sorted]
part = [v/len(questions) for v in values + [something_else]]

plt.bar(range(len(count) + 1), part, align='center')
plt.xticks(range(len(count) + 1), keys_sorted + ['something else'])
plt.show()