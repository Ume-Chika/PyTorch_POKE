import os
import json
import requests

from tqdm import tqdm

"""API 呼び出し先"""
ep1 = 'https://pokeapi.co/api/v2/pokemon-species/{id or name}/' # エンドポイント1
ep2 = 'https://pokeapi.co/api/v2/pokemon/{id or name}/' # エンドポイント2
ep_img = 'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{id or name}.png' # エンドポイント for image

def get_generation_num(gen_str):
    gen_changer = {
        'i' : 1,
        'ii': 2,
        'iii': 3,
        'iv': 4,
        'v': 5,
        'vi': 6,
        'vii': 7,
        'viii': 8,
        'ix': 9
    }
    return gen_changer.get(gen_str.split('-')[1], None)

def get_poke_data_from_json(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, 'r', encoding='utf-8') as f: # 読み込みモードを'r'に、encodingを指定
        try:
            poke_data = json.load(f)
        except json.JSONDecodeError:
            print(f"警告: '{json_path}' が不正なJSON形式です。空のデータとして扱います。")
            poke_data = {}
    return poke_data

def write_poke_dict_to_json(json_path, new_poke_dict):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(new_poke_dict, f, ensure_ascii=False, indent=4)

# def save_image_as_png(image_obj, output_path):
#     """
#     PIL Image オブジェクトをPNGファイルとして保存します。
#     """
#     try:
#         image_obj.save(output_path, format="PNG")
#         print(f"画像を '{output_path}' に保存しました。")
#         return True
#     except Exception as e:
#         print(f"画像の保存中にエラーが発生しました: {e}")
#         return False

def get_poke_data(id):
    try:
        # ep1から情報を取得
        response = requests.get(ep1.replace("{id or name}", str(id)))
        poke_info = response.json()
        for names_dict in poke_info["names"]:
            if names_dict['language']['name'] == 'ja':
                poke_name = (names_dict['name'])
                break
        poke_gen = get_generation_num(poke_info['generation']['name'])

        # ep2から情報を取得
        response = requests.get(ep2.replace("{id or name}", str(id)))
        poke_info = response.json()
        poke_types = [poke_types['type']['name'] for poke_types in poke_info['types']]
        poke_state = {poke_stat['stat']['name']: poke_stat['base_stat'] for poke_stat in poke_info['stats']}
        result_dict = {
            'id':id,
            'name':poke_name,
            'type':poke_types,
            'generation':poke_gen,
            'state': poke_state,
        }
        return result_dict
    except requests.exceptions.RequestException as e:
        print(f"id = {id}: リクエスト中にエラーが発生しました: {e}")
        return None

def get_and_save_poke_img(id, image_path, img_show=False):
    # ファイルパスからディレクトリ部分を抽出
    output_dir = os.path.dirname(image_path)
    if output_dir: # output_dir が空文字列でないことを確認
        os.makedirs(output_dir, exist_ok=True)

    try:
        # ep_imgから画像を取得, Tensorへ変換
        response = requests.get(ep_img.replace("{id or name}", str(id)))
        if response.status_code == 200: # 正常なレスポンス
            with open(image_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f'Faild to dwonload poke_{id}. Status code = {response.status_code}')

    except requests.exceptions.RequestException as e:
        print(f"id = {id}: リクエスト中にエラーが発生しました: {e}")
        return None

if __name__ == "__main__":
    json_path = 'poke_data.json'
    """ ポケモン1025種のデータと画像をダウンロード
    ・data：データ
        dataは辞書型, キーにid(文字列)を与えると各ポケモンの辞書にアクセスする
        各ポケモンの辞書は次の情報を含む
            key = 'id', value = ポケモンのid(整数)
            key = 'name', value = ポケモンの名前の文字列
            key = 'type', value = ポケモンのタイプ(文字列)が最大2つ格納されたリスト
            key = 'generation', value = ポケモンの世代(整数)
            key = 'state', value = 種族値が格納された辞書
    ・img：画像
        imgはpng形式で保存される
    """
    poke_data = get_poke_data_from_json(json_path) # すでに保存済みのデータ
    print('get poke info from POKEAPI')
    for id in tqdm(range(1, 1025+1)):
        if not str(id) in poke_data.keys():
            single_poke_data = get_poke_data(id)
            poke_data[str(id)] = single_poke_data
    write_poke_dict_to_json(json_path, poke_data)
    print('complete!!')

    print('get poke image from POKEAPI')
    for id in tqdm(range(1, 1025+1)):
        image_path = f'poke_png/poke_{id}.png'
        if not os.path.exists(image_path):
            get_and_save_poke_img(id, image_path)

    print('complete!!')