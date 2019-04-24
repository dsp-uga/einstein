from einstein.utils.summary_table import draw


def test_draw():
    year = '2017'
    target_hr = '1'
    grid = '1'
    model_name = 'abc'
    dictionary = { 'a': 1, 'b': 2}
    output = draw(year, target_hr, grid, model_name, dictionary)
    # Testing that there is atleast something which is being printed
    assert len(str(output)) > 0

