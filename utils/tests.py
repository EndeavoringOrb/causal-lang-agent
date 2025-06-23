from utils.utils import is_correct

def test_is_correct():
    data = {
        "answer": "treatment group",
        "meta_data": {
            "question_type": "multiple_choice",
        },
    }
    assert is_correct("treatment group", data) == True
    assert is_correct(" treatment group", data) == True
    assert is_correct(" treatment group.", data) == True
    assert is_correct("treatment group}", data) == True
