stations = [
    "二子玉川", "上野毛", "等々力", "尾山台", "九品仏", 
    "自由が丘", 
    "緑が丘", "大岡山"
]
express_stops = ["二子玉川", "自由が丘", "大岡山"]
destination = "緑が丘"

start_idx = stations.index("自由が丘")
next_express_station = None

for i in range(start_idx - 1, -1, -1):
    if stations[i] in express_stops:
        next_express_station = stations[i]
        break

current_idx = stations.index(next_express_station)
dest_idx = stations.index(destination)

stations_count = dest_idx - current_idx

print(f"目的地までの駅数: {stations_count}駅先")