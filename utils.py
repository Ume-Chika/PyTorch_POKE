type_to_idx = {
    'normal': 0, #（ノーマル）
    'fire'  : 1, #（ほのお） 
    'water' : 2, #（みず）
    'grass' : 3, #（くさ）
    'electric':4,#（でんき）
    'ice'   : 5, #（こおり）
    'fighting':6,#（かくとう）
    'poison': 7, #（どく）
    'ground': 8, #（じめん） 
    'flying': 9, #（ひこう）
    'psychic':10,#（エスパー）
    'bug'   : 11,#（むし） 
    'rock'  :12, #（いわ）
    'ghost' :13, #（ゴースト）
    'dragon':14, #（ドラゴン）
    'steel' :15, #（はがね）
    'dark'  :16, #（あく）
    'fairy' :17, #（フェアリー）
}
idx_to_type = {v: k for k, v in type_to_idx.items()}