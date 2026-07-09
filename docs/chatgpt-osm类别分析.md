详细解释一下下面出现的所有类别的含义，osm的类别，哪些是谁跑的什么特征（比如，车，人，等）：

============================================================
【DelvMap road 大图】收的 highway 类型 (generate_porto_delvmap_bigimgs.py)
============================================================
总 way: 15343
  residential: 8590 (56.0%)
  pedestrian: 1915 (12.5%)
  unclassified: 985 (6.4%)
  secondary: 792 (5.2%)
  primary: 772 (5.0%)
  motorway_link: 477 (3.1%)
  tertiary: 415 (2.7%)
  track: 375 (2.4%)
  motorway: 372 (2.4%)
  primary_link: 157 (1.0%)
  road: 147 (1.0%)
  trunk_link: 79 (0.5%)
  secondary_link: 73 (0.5%)
  living_street: 73 (0.5%)
  tertiary_link: 58 (0.4%)
  trunk: 51 (0.3%)
  services: 4 (0.0%)
  construction: 2 (0.0%)
  proposed: 2 (0.0%)
  bus_stop: 1 (0.0%)
  platform: 1 (0.0%)
  elevator: 1 (0.0%)
  raceway: 1 (0.0%)

其中闭合多边形(首尾同节点, 会画成闭合): {'pedestrian': 1692, 'residential': 116, 'primary': 46, 'secondary': 42, 'tertiary': 29, 'unclassified': 23, 'track': 10, 'tertiary_link': 5, 'services': 4, 'primary_link': 3}

============================================================
【samroad GT pickle】收的 way 类型 (download_use_osm.py OSMHandler, 无过滤)
============================================================
总 way: 44182
  building: 15871 (35.9%)
  highway=residential: 8590 (19.4%)
  highway=service: 2256 (5.1%)
  highway=pedestrian: 1915 (4.3%)
  barrier: 1722 (3.9%)
  landuse: 1439 (3.3%)
  leisure: 1354 (3.1%)
  amenity: 1167 (2.6%)
  highway=unclassified: 985 (2.2%)
  addr:city: 811 (1.8%)
  highway=secondary: 792 (1.8%)
  highway=primary: 772 (1.7%)
  highway=steps: 720 (1.6%)
  highway=footway: 647 (1.5%)
  highway=motorway_link: 477 (1.1%)
  highway=tertiary: 415 (0.9%)
  highway=track: 375 (0.8%)
  highway=motorway: 372 (0.8%)
  electrified: 321 (0.7%)
  (no_tags): 294 (0.7%)
  area: 266 (0.6%)
  addr:street: 242 (0.5%)
  natural: 178 (0.4%)
  highway=primary_link: 157 (0.4%)
  addr:housename: 151 (0.3%)
  gauge: 148 (0.3%)
  highway=road: 147 (0.3%)
  name: 124 (0.3%)
  highway=path: 122 (0.3%)
  access: 119 (0.3%)

其中闭合多边形: {'building': 15870, 'highway=pedestrian': 1692, 'landuse': 1437, 'leisure': 1353, 'amenity': 1167, 'addr:city': 811, 'area': 266, 'addr:street': 242, '(no_tags)': 215, 'addr:housename': 151}

# Answer

你的这两个统计实际上对应 **OpenStreetMap (OSM) 的 Tag 系统**。

OSM 并不是像 COCO 那样定义"汽车、人、自行车"类别，而是给地图中的每一个对象（Node、Way、Relation）打 **Key=Value** 标签，例如：

```
highway=residential
building=yes
landuse=residential
amenity=school
```

所以你的统计里的类别其实混合了：

* 道路（highway=*）
* 建筑物（building）
* 土地利用（landuse）
* 兴趣点（amenity）
* 地址信息（addr:*）
* 铁路属性（electrified）
* 自然地物（natural）
* 障碍物（barrier）
* ...

下面详细解释。

---

# 一、OSM对象类型

OSM只有三种几何对象：

| 对象       | 含义                   |
| -------- | -------------------- |
| Node     | 一个点，例如公交站、路灯、树、人行横道点 |
| Way      | 一串点，可以表示道路，也可以表示建筑轮廓 |
| Relation | 多个Way组成，例如公交线路、行政区   |

你的统计：

```
总 way: 44182
```

说明这里只统计了 **Way**。

---

# 二、highway=* 不是"高速公路"

很多人容易误解。

OSM中

```
highway=*
```

表示

> **所有交通通行路线（Transportation network）**

包括：

* 高速公路
* 城市道路
* 小路
* 人行道
* 台阶
* 自行车道

全部属于 highway。

所以 highway 实际可以理解为：

> Transportation corridor

而不是中文里的"高速公路"。

---

# 三、你的 DelvMap 收集的 highway 类别

下面逐个解释。

---

## 1. residential

```
highway=residential
```

居民区道路。

典型：

* 小区道路
* 城市住宅街道
* 双向两车道
* 可以汽车通行

例如

```
居民区
│
├──住宅
├──住宅
└──────
```

主要服务：

✔ 汽车

✔ 自行车

✔ 行人

---

## 2. pedestrian

```
highway=pedestrian
```

步行街。

例如：

商业街

广场

旅游街

不能正常机动车通行。

服务对象：

✔ 行人

部分允许：

* 自行车
* 配送车辆

例如：

```
南京路
王府井
```

---

## 3. unclassified

很多人误解。

不是

> 未分类

而是

> 最低等级公共道路

例如：

农村连接道路

小镇主路

比 residential 更偏连接功能。

可以汽车。

---

## 4. secondary

二级道路。

例如：

城市主干道。

连接：

城区

商业区

居民区

汽车主行驶道路。

---

## 5. primary

一级道路。

城市快速连接道路。

例如：

城市主干路

国道

省道

交通量较大。

汽车。

---

## 6. motorway

真正高速公路。

例如：

高速

Expressway

Freeway

特点：

* 禁止行人
* 禁止自行车
* 高速机动车

---

## 7. motorway_link

高速匝道。

例如：

```
=========
      \
       \
=========
```

高速上下匝道。

---

## 8. trunk

干线公路。

介于：

primary

和

motorway

之间。

很多国家：

快速路。

---

## 9. trunk_link

干线匝道。

---

## 10. primary_link

一级道路匝道。

连接：

primary

与其它道路。

---

## 11. secondary_link

二级道路连接线。

---

## 12. tertiary

三级道路。

比 secondary 小。

例如：

城区支路。

---

## 13. tertiary_link

三级道路连接。

---

## 14. track

农用道路。

林道。

泥土路。

例如：

```
农田

=====
```

一般：

拖拉机

越野车

农业车辆。

---

## 15. road

未知道路。

OSM里：

```
不知道具体等级
```

以后再修改。

临时标签。

---

## 16. living_street

生活街。

特点：

汽车让行人。

限速：

20 km/h

儿童可以玩。

欧洲很多。

---

## 17. services

服务区道路。

例如：

高速服务区

停车场内部道路

加油站内部道路

不是普通城市道路。

---

## 18. construction

建设中的道路。

尚未通车。

---

## 19. proposed

规划道路。

地图已有规划。

现实不存在。

---

## 20. bus_stop

公交站。

严格来说：

一般应是 Node。

这里说明有人画成 Way。

---

## 21. platform

站台。

公交站台

火车站台

地铁站台

可以是面。

---

## 22. elevator

电梯。

例如：

地下通道电梯。

---

## 23. raceway

赛车场。

例如：

F1

赛车场道路。

---

# 四、GT pickle里的其它类别

这些已经不是 highway。

---

## building

建筑物。

包括：

住宅

学校

医院

商场

办公楼

通常：

闭合 polygon。

所以你看到：

```
building
15870 个闭合
```

几乎全部都是面。

---

## barrier

障碍物。

例如：

围栏

墙

护栏

栅栏

路桩

---

## landuse

土地利用。

例如：

住宅区

工业区

森林

农田

商业区

不是道路。

---

## leisure

休闲设施。

例如：

公园

操场

高尔夫

体育场

游乐园

---

## amenity

公共设施。

例如：

学校

医院

停车场

厕所

银行

邮局

餐馆

加油站

---

## natural

自然地物。

例如：

树林

湖泊

湿地

海岸

沙滩

岩石

---

# 五、其它 highway

GT比DelvMap多了一些。

---

## service

```
highway=service
```

服务道路。

例如：

停车场内部

小区内部

加油站

学校内部

医院内部

汽车可以。

但不是公共道路。

---

## steps

```
highway=steps
```

楼梯。

例如：

地下通道楼梯。

只能：

行人。

---

## footway

```
highway=footway
```

人行道。

例如：

```
██████
------
```

不能汽车。

---

## path

```
highway=path
```

通用小路。

可能：

步行

骑车

骑马

山区。

用途未知。

---

# 六、铁路属性

## electrified

铁路是否电气化。

通常：

```
railway=rail
electrified=yes
```

这里只统计到了 key。

---

## gauge

铁路轨距。

例如：

1435 mm

1000 mm

---

# 七、地址信息

这些其实不是道路。

只是建筑上的标签。

例如：

```
addr:city
```

城市。

```
addr:street
```

街道名。

```
addr:housename
```

楼名。

---

# 八、其它

## access

通行权限。

例如：

```
private

destination

yes

no
```

---

## area

说明：

这是一个面。

不是线。

---

## name

名称。

例如：

```
人民路
```

---

## (no_tags)

没有任何标签。

OSM存在这种情况。

可能：

Relation的一部分。

或者数据不完整。

---

# 九、闭合多边形为什么 highway 也有？

例如：

```
pedestrian
1692
```

很多人疑惑。

原因：

OSM允许

```
highway=pedestrian
```

画成一个广场。

例如：

```
□□□□□□
□      □
□      □
□□□□□□
```

整个步行广场。

不是一条线。

所以：

```
首尾节点相同
```

形成 polygon。

例如：

欧洲广场。

商业中心。

同样：

```
residential
```

偶尔也有人画成闭合环路。

---

# 十、如果按交通参与者分类

对于你的地图补全（Map Completion）任务，更有意义的是按**交通对象**来理解 `highway=*` 的服务对象：

| highway 类型         |      汽车     |   行人  |    自行车    | 典型场景            |
| ------------------ | :---------: | :---: | :-------: | --------------- |
| motorway           |      ✅      |   ❌   |     ❌     | 高速公路            |
| motorway_link      |      ✅      |   ❌   |     ❌     | 高速匝道            |
| trunk              |      ✅      | ❌（通常） |   ❌（通常）   | 城市快速路/干线        |
| trunk_link         |      ✅      |   ❌   |     ❌     | 干线匝道            |
| primary            |      ✅      |   ⚠️  |     ⚠️    | 一级主干道           |
| primary_link       |      ✅      |   ⚠️  |     ⚠️    | 一级道路连接线         |
| secondary          |      ✅      |   ✅   |     ✅     | 二级主干道           |
| secondary_link     |      ✅      |   ✅   |     ✅     | 二级道路连接线         |
| tertiary           |      ✅      |   ✅   |     ✅     | 支路              |
| tertiary_link      |      ✅      |   ✅   |     ✅     | 支路连接线           |
| residential        |      ✅      |   ✅   |     ✅     | 居民区道路           |
| living_street      |    ✅（低速）    | ✅（优先） |     ✅     | 生活街区            |
| service / services |      ✅      |   ✅   |     ✅     | 停车场、医院、加油站等内部道路 |
| unclassified       |      ✅      |   ✅   |     ✅     | 低等级公共道路         |
| road               |   ✅（等级未知）   |   ✅   |     ✅     | 未细分类道路          |
| track              | ⚠️（农业/越野车辆） |   ✅   |     ✅     | 农田、林区土路         |
| pedestrian         |    ❌（通常）    |   ✅   | ⚠️（视当地规定） | 步行街、广场          |
| footway            |      ❌      |   ✅   |   ❌（通常）   | 人行道             |
| steps              |      ❌      |   ✅   |     ❌     | 楼梯              |
| path               |    ❌（通常）    |   ✅   |   ✅（可能）   | 山路、小径、共享步道      |
| bus_stop           |    公交停靠设施   |   ✅   |     —     | 公交站             |
| platform           |      站台     |   ✅   |     —     | 公交/地铁/火车站台      |
| elevator           |      电梯     |   ✅   |     —     | 无障碍设施           |
| raceway            |     专用赛车    |   ❌   |     ❌     | 赛车赛道            |
| construction       |   🚧（未开放）   |   🚧  |     🚧    | 建设中的道路          |
| proposed           |    ❌（规划中）   |   ❌   |     ❌     | 尚未建设            |

对于 **DelvMap / SAM-Road** 这类基于遥感影像的道路提取或地图补全任务，通常会保留所有具有道路几何形态的 `highway=*` 对象（如 `residential`、`primary`、`secondary`、`motorway` 等），而像 `footway`、`steps`、`pedestrian` 是否纳入，则取决于数据集希望学习的是**机动车路网**还是**完整交通网络**。从你给出的统计来看，DelvMap 保留了 `pedestrian` 和少量特殊道路，但过滤掉了 `footway`、`steps`、`service` 等较细粒度的通行设施，因此其目标更接近**可见道路网络**而不是完整的 OSM 交通基础设施。
