stations = [
    "二子玉川", "上野毛", "等々力", "尾山台", "九品仏", 
    "自由が丘", 
    "緑が丘", "大岡山", "北千束", "旗の台"
]

express_stops = ["二子玉川", "自由が丘", "大岡山", "旗の台"]


start_idx = stations.index("自由が丘")

next_express_station = None
for i in range(start_idx + 1, len(stations)):
    if stations[i] in express_stops:
        next_express_station = stations[i]
        break

dropped_idx = stations.index(next_express_station)
destination_idx = dropped_idx - 1
destination_station = stations[destination_idx]

print(f"目的地: {destination_station}")