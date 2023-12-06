import json

from data.logiqav2 import _format_option_list


class ARLSATReader:
    rank2option = ['A', 'B', 'C', 'D', 'E']

    def __init__(self, flat_options: bool = False, option_order: str = "ABCDE"):
        self.flat_options = flat_options
        self.option_order = option_order

    def __call__(self, file):
        all_context = []
        all_question = []
        all_option_list = []
        all_label = []
        print(file)
        data = json.load(open(file, "r"))
        for item in data:
            for q in item["questions"]:
                all_context.append(item["passage"])
                all_question.append(q["question"])

                options = []
                ordered_label = -1
                for i, x in enumerate(self.option_order):
                    idx = ord(x) - ord('A')
                    options.append(q["options"][idx])

                    if x == q["answer"]:
                        ordered_label = i

                if "Test" not in file:
                    assert ordered_label != -1

                all_label.append(ordered_label)
                all_option_list.append(options)

        return [
            {
                "context": context,
                "question": question,
                "option_list": _format_option_list(option_list, self.rank2option) if self.flat_options else option_list,
                "label": label,
            } for context, question, option_list, label in zip(all_context, all_question, all_option_list, all_label)
        ]
