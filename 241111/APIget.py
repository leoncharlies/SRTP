import requests

# 高德地图API密钥
apiKey = '56ca0f236d83240e3454bdc44017e7f0'

# 请求道路拥堵状况
def get_traffic_status(lat, lon):
    # 设置一个小矩形范围以包含目标经纬度附近的道路
    delta = 0.001  # 设置精度范围
    rectangle = f'{lon - delta},{lat - delta};{lon + delta},{lat + delta}'

    # 发起API请求
    url = f'https://restapi.amap.com/v3/traffic/status/rectangle?rectangle={rectangle}&key={apiKey}'
    try:
        response = requests.get(url)
        response.raise_for_status() 

        data = response.json()

        print("返回的数据结构: ", data)

        if data['status'] == '1':  # 查询成功
            print("拥堵状况查询成功")

            if 'trafficinfo' in data and 'roads' in data['trafficinfo']:
                for road in data['trafficinfo']['roads']:
                    print(f"道路名称: {road['name']}")
                    print(f"道路状态: {road['status']}")  # 1-畅通, 2-缓行, 3-拥堵, 4-严重拥堵
                    print(f"速度: {road['speed']} km/h")
            else:
                print("没有道路数据返回.")
        else:
            print(f"查询失败: {data['info']}")
    except requests.exceptions.RequestException as e:
        print(f"请求出错: {e}")

get_traffic_status(30.609455, 104.030356)
