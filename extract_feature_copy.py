import json
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
import xmltodict
import numpy as np
import collections
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import multiprocessing


def get_outer_wall(dictionary):
    """
    get the outer wall data
    :param dictionary: the dict, no root
    :return: a dataFrame contain all outerWall_info, and add one column whether it is vertical or horizontal
    """
    wall_df = pd.DataFrame(dictionary['WallInfo']['WallData'])

    wall_version = dictionary['WallInfo']['@Version']

    wall_df[['@StartX', '@StartY', '@EndX', '@EndY', '@Thickness', '@Height']] = wall_df[['@StartX', '@StartY', '@EndX', '@EndY', '@Thickness', '@Height']].astype(float)
    wall_df = wall_df[['@StartX', '@StartY', '@EndX', '@EndY', '@Thickness', '@Height']]

    if wall_version == '2_2':
        # StartX = lambda value: (value - 4000) * 20
        # StartY = lambda value: -(value - 4000) * 20
        # EndX = lambda value: (value - 4000) * 20
        # EndY = lambda value: -(value - 4000) * 20
        # Thickness = lambda value:  value * 20
        # Height = lambda value:  value * 10
        StartX = lambda value: round(value / 20 + 4000, 2)
        StartY = lambda value: round(-(value / 20) + 4000, 2)
        EndX = lambda value: round(value / 20 + 4000, 2)
        EndY = lambda value: round(-(value / 20) + 4000, 2)
        Thickness = lambda value: round(value / 20, 2)
        Height = lambda value: round(value / 20, 2)
        wall_df.loc[:, '@StartX'] = wall_df.loc[:, '@StartX'].apply(StartX)
        wall_df.loc[:, '@StartY'] = wall_df.loc[:, '@StartY'].apply(StartY)
        wall_df.loc[:, '@EndX'] = wall_df.loc[:, '@EndX'].apply(EndX)
        wall_df.loc[:, '@EndY'] = wall_df.loc[:, '@EndY'].apply(EndY)
        wall_df.loc[:, '@Height'] = wall_df.loc[:, '@Height'].apply(Height)
        wall_df.loc[:, '@Thickness'] = wall_df.loc[:, '@Thickness'].apply(Thickness)
    elif wall_version == '2_1':
        wall_df = wall_df

    for index, row in wall_df.iterrows():
        if abs(row['@StartX'] - row['@EndX']) < 5:
            wall_df.loc[index, 'layout'] = 'V'
        elif abs(row['@StartY'] - row['@EndY']) < 5:
            wall_df.loc[index, 'layout'] = 'H'
    return wall_df


def get_inner_wall(dictionary):
    """
    get the inner wall data
    :param: dictionary: the dict, no root
    :return: a dataFrame contain all innerWall_info, and add one column whether it is vertical or horizontal
    """
    inner_wall_df = pd.DataFrame(dictionary['InnerWallInfo']['InnerWallData'])
    inner_wall_df[['@RoomID', '@StartX', '@StartY', '@EndX', '@EndY', '@Height3D']] = inner_wall_df[
        ['@RoomID', '@StartX', '@StartY', '@EndX', '@EndY', '@Height3D']].astype(float)

    inner_wall_df = inner_wall_df[['@RoomID', '@StartX', '@StartY', '@EndX', '@EndY', '@Height3D']]

    round_2 = lambda x: round(x, 2)
    inner_wall_df[['@StartX', '@StartY', '@EndX', '@EndY', '@Height3D']] = inner_wall_df[
        ['@StartX', '@StartY', '@EndX', '@EndY', '@Height3D']].apply(round_2)

    # 添加水平垂直属性
    for index, row in inner_wall_df.iterrows():
        # if row['@StartX'] == row['@EndX']:
        if abs(row['@StartX'] - row['@EndX']) < 5:
            inner_wall_df.loc[index, 'layout'] = 'V'
        # elif row['@StartY'] == row['@EndY']:
        elif abs(row['@StartY'] - row['@EndY']) < 5:
            inner_wall_df.loc[index, 'layout'] = 'H'
    return inner_wall_df


def get_dist(pos1, pos2, query_pos):

    x1 = pos1[0]
    y1 = pos1[1]
    x2 = pos2[0]
    y2 = pos2[1]
    x3 = query_pos[0]
    y3= query_pos[1]
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    d = abs(a * x3 + b * y3 + c) / np.sqrt(pow(a, 2) + pow(b, 2))

    return round(d, 1)


def process_door(dictionary):
    """
    get the doors info, and add one column (horizontal or vertical)
    :param dictionary: the dict from the json file, no root
    :return:
        door_df: a dataFrame of doors info
        door_room_dict: a dict { wallID(outerwall): [RoomID, RoomID](one door could belongs to two room}
    """

    inner_wall_df = get_inner_wall(dictionary)
    outer_wall_df = get_outer_wall(dictionary)

    door_data = dictionary['InWallInfo']['InWallData']
    door_version = dictionary['InWallInfo']['@Version']

    if len(door_data) == 1:
        door_df = pd.Series(door_data).to_frame()
    else:
        door_df = pd.DataFrame(door_data)
    # print(door_df)
    # 从中提取门的信息，转换数据类型
    door_df[['@PosX', '@PosY', '@Length', '@Width', '@Height',
             '@RotateZ', '@WallID']] = door_df[['@PosX', '@PosY', '@Length', '@Width', '@Height',
                                               '@RotateZ','@WallID']].astype(float)
    door_df = door_df[['@PosX', '@PosY', '@Length', '@Width', '@Height',
                       '@RotateZ','@WallID', '@MaterialID', '@Type']]

    door_df.loc[:, '@Type'][door_df['@Type'] == ''] = 'DOOR'

    if door_version == '2_2':
        PosX = lambda value: round(value / 20 + 4000, 2)
        PosY = lambda value: round(-(value / 20) + 4000, 2)
    elif door_version == '2_1':
        PosX = lambda value: round(value / 2 + 4000, 2)
        PosY = lambda value: round(-(value / 2) + 4000, 2)

    door_df.loc[:, 'X'] = door_df['@PosX'].apply(PosX)
    door_df.loc[:, 'Y'] = door_df['@PosY'].apply(PosY)

    # 可以转换角度
    # print(outer_wall_df)

    for index, row in door_df.iterrows():
        door_df.loc[index, 'layout'] = str(outer_wall_df.loc[int(row['@WallID']), 'layout'])

    for index, row in door_df.iterrows():
        door_df.loc[index, 'doorID'] = str(index)
    # 生成门和对应房间的字典
    room_door_dict = {}
    for index, row in door_df[door_df['@Type'] == 'DOOR'].iterrows():

        dist_dict = collections.defaultdict(list)
        X = row['X']
        Y = row['Y']

        if row['layout'] == 'H':

            door_df.loc[index, '@StartX'] = X - row["@Length"] / 40
            door_df.loc[index, '@StartY'] = Y
            door_df.loc[index, '@EndX'] = X + row["@Length"] / 40
            door_df.loc[index, '@EndY'] = Y

            for wall_index, wall_row in inner_wall_df[inner_wall_df['layout'] == 'H'].iterrows():
                if (wall_row['@StartX'] < row['X'] < wall_row['@EndX']) or (
                        wall_row['@StartX'] > row['X'] > wall_row['@EndX']):
                    dist = get_dist([wall_row['@StartX'], wall_row['@StartY']],
                                    [wall_row['@EndX'], wall_row['@EndY']],
                                    [row['X'], row['Y']])
                    dist_dict['dist'].append(dist)
                    dist_dict['room_id'].append(wall_row['@RoomID'])
                    dist_dict['WallID'].append(row['doorID'])
        if row['layout'] == 'V':

            door_df.loc[index, '@StartY'] = Y - float(row["@Length"]) / 40

            door_df.loc[index, '@StartX'] = X
            door_df.loc[index, '@EndY'] = Y + float(row["@Length"]) / 40
            door_df.loc[index, '@EndX'] = X

            for wall_index, wall_row in inner_wall_df[inner_wall_df['layout'] == 'V'].iterrows():
                if (wall_row['@StartY'] < row['Y'] < wall_row['@EndY']) or (
                        wall_row['@StartY'] > row['Y'] > wall_row['@EndY']):
                    dist = get_dist([wall_row['@StartX'], wall_row['@StartY']],
                                    [wall_row['@EndX'], wall_row['@EndY']],
                                    [row['X'], row['Y']])
                    dist_dict['dist'].append(dist)
                    dist_dict['room_id'].append(wall_row['@RoomID'])
                    dist_dict['WallID'].append(row['doorID'])

        dist_df = pd.DataFrame(dist_dict)

        # dist_df = dist_df[dist_df['dist'] == min(dist_df['dist'].values)]
        dist_df = dist_df[dist_df['dist'] == min(dist_df['dist'].values)]
        room_door_dict[dist_df['WallID'].values[0]] = list(set(dist_df['room_id'].values))

    return door_df, room_door_dict


def process_out_wall(out_wall_df, inner_wall_df):

    wall_df_copy = out_wall_df

    wall_df_hor = wall_df_copy[wall_df_copy['layout'] == 'H']
    wall_df_ver = wall_df_copy[wall_df_copy['layout'] == 'V']
    wall_list = []

    for index, row in inner_wall_df.iterrows():
        layout = row['layout']
        min_x = min(row['@StartX'], row['@EndX'])
        min_y = min(row['@StartY'], row['@EndY'])

        max_x = max(row['@StartX'], row['@EndX'])
        max_y = max(row['@StartY'], row['@EndY'])

        if layout == 'V':
            for index_out, row_out in wall_df_ver.iterrows():
                thick = row_out['@Thickness'] / 2
                dist = abs(row['@StartX'] - row_out['@StartX'])
                if np.abs(dist -thick)<0.01:
                    if (((row_out['@StartY'] >= min_y) and (row_out['@StartY'] <= max_y))or ((row_out['@EndY'] >= min_y) and (row_out['@EndY'] <= max_y)))\
                            or (((row_out['@StartY'] <= min_y) and (row_out['@EndY'] >= max_y))or ((row_out['@StartY'] <= min_y) and (row_out['@EndY'] >= max_y))):
                        wall_list.append(index_out)

        if layout == 'H':
            for index_out, row_out in wall_df_hor.iterrows():
                thick = row_out['@Thickness'] / 2
                dist = abs(row['@StartY'] - row_out['@StartY'])
                if np.abs(dist - thick) < 0.01:
                    if (((row_out['@StartX'] >= min_x) and (row_out['@StartX'] <= max_x)) or ((row_out['@EndX'] >= min_x) and (row_out['@EndX'] <= max_x)))\
                            or (((row_out['@StartX'] <= min_x) and (row_out['@EndX'] >= max_x)) or ((row_out['@EndX'] <= min_x) and (row_out['@StartX'] >= max_x))):

                        wall_list.append(index_out)

    wall_df_copy = wall_df_copy[wall_df_copy.index.isin(wall_list)]
    # print(inner_wall_df)
    # print(wall_df_copy)
    # for index, row in wall_df_copy.iterrows():
    #     plt.plot((row['@StartX'], row['@EndX']), (row['@StartY'], row['@EndY']))
    #     plt.annotate(index, xy=((row['@StartX'] + row['@EndX']) / 2, (row['@StartY'] + row['@EndY']) / 2), xycoords='data', xytext=(+30, -30),
    #                  textcoords='offset points', fontsize=13, arrowprops=dict(arrowstyle='->',
    #                                                                           connectionstyle="arc3,rad=.2", color='black'))
    #
    # for index, row in inner_wall_df.iterrows():
    #     plt.plot((row['@StartX'], row['@EndX']), (row['@StartY'], row['@EndY']))
    #     plt.annotate(index, xy=((row['@StartX'] + row['@EndX']) / 2, (row['@StartY'] + row['@EndY']) / 2),
    #                  xycoords='data', xytext=(+30, -30),
    #                  textcoords='offset points', fontsize=13, arrowprops=dict(arrowstyle='->',
    #                                                                           connectionstyle="arc3,rad=.2"), color='r')
    # plt.show()
    return wall_df_copy


def get_room(dictionary):
    """
    得到所有房间的描述信息，
    提取房间的面积，
    将@RoomID与@RoomName对应起来，
    :param dictionary:
    :return:
    """

    room_df = pd.DataFrame(dictionary['RoomInfo']['RoomData'])
    room_df[['@RoomSize', '@RoomId']] = room_df[['@RoomSize', '@RoomId']].astype(float)

    room_df = room_df[['@RoomName', '@RoomSize', '@RoomId']]

    for index, row in room_df.iterrows():
        room_df.loc[index, '@RoomID'] = index

    return room_df


def min_max_xy(inner_wall_df):
    """
    get the minimal and maximal coordinators of every room
    :param dictionary: the dict, no root
    :return: a dict {roomID:{'min_x': value, 'min_y': value, 'max_x': value, 'max_y': value}, ... }
    """

    min_x = min(min(inner_wall_df['@StartX']), min(inner_wall_df['@EndX']))

    min_y = min(min(inner_wall_df['@StartY']), min(inner_wall_df['@EndY']))
    max_x = max(max(inner_wall_df['@StartX']), max(inner_wall_df['@EndX']))
    max_y = max(max(inner_wall_df['@StartY']), max(inner_wall_df['@EndY']))
    xy_min_max_dict = {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}

    return xy_min_max_dict


def rotate(theta, center_position, position):
    """
    旋转房间
    :param theta:
    :param center_position:
    :param position:
    :return:result_dict
    """
    result_dict = {}
    theta = np.pi * (theta / 180)
    c = np.cos(theta)
    s = np.sin(theta)
    Ox_ = (1 - c) * center_position[0] + s * center_position[1]
    Oy_ = (1 - c) * center_position[1] - s * center_position[0]
    rotato_mat = np.array([
        [c, -s, Ox_],
        [s, c, Oy_],
        [0, 0, 1]
    ])

    position.append(1)
    position = np.array(position)

    result = np.dot(rotato_mat, position)
    result_dict['x'] = result[0]
    result_dict['y'] = result[1]
    return result_dict


def get_furniture(dictionary):
    """
    get all furniture of the whole house, drop some unnecessary attributes
    :param dictionary: the xml dictionary
    :return: furniture dataframe
    """
    furniture_data = dict(dictionary['FurnitureInfo'])['FurnitureData']
    furniture_df = pd.DataFrame(furniture_data)

    furniture_df[['@PosX', '@PosY', '@PosZ', '@Length', '@Width', '@Height', '@RotateX', '@RotateY', '@RotateZ']] = \
        furniture_df[['@PosX', '@PosY', '@PosZ', '@Length', '@Width', '@Height', '@RotateX', '@RotateY', '@RotateZ']].astype(float)
    furniture_df['@RoomID'].astype(int)
    furniture_df = furniture_df[['@PosX', '@PosY', '@PosZ', '@Length', '@Width', '@Height', '@RotateX',
                  '@RotateY', '@RotateZ', '@RoomID',  '@MaterialID']]
    return furniture_df


def get_material_info(dictionary):
    """
    get all get_material info, drop some unnecessary attributes
    :param dictionary:
    :return:
    """
    material_df = pd.DataFrame(dictionary)
    material_df = material_df[['MaterialId', 'MaterialName', 'CategoryId', 'PlaceRule', 'MaterialType']]  # drop many useless data
    material_df = material_df.rename(columns={'MaterialId': '@MaterialID'})

    return material_df


def get_room_furniture(furniture_df, room_id):
    """
    获得单个房间的家具, 给每个家具分区
    :param furniture_df: the furniture dataframe for all rooms
    :param room_id:
    :return: all furniture in this room
    """

    furniture_df = furniture_df[furniture_df['@RoomID'] == room_id]
    if not furniture_df.shape[0]:
        return None
    x_norm = lambda x: float(x) / 2 + 4000
    y_norm = lambda y: -float(y) / 2 + 4000
    furniture_df['X'] = furniture_df['@PosX'].apply(x_norm)
    furniture_df['Y'] = furniture_df['@PosY'].apply(y_norm)

    # 根据家具名字提取功能区(后期需要改进）
    for index, row in furniture_df.iterrows():
        if '床' in row['MaterialName']:
            furniture_df.loc[index, 'function'] = '睡眠区'
        elif '衣柜' in row['MaterialName'] or '收纳' in row['MaterialName']:
            furniture_df.loc[index, 'function'] = '衣柜区'
        elif '妆' in row['MaterialName']:
            furniture_df.loc[index, 'function'] = '化妆区'
        elif '电视' in row['MaterialName']:
            furniture_df.loc[index, 'function'] = '影音区'
        elif '字' in row['MaterialName']:
            furniture_df.loc[index, 'function'] = '办公区'
        # 角度换算
        theta = round((row['@RotateZ'] * 180) / np.pi, 1)
        while theta < 0:
            theta = theta + 360
        furniture_df.loc[index, 'Theta'] = theta

        x = row['X']
        y = row['Y']
        l = row['@Length'] / 40
        w = row['@Width'] / 40
        if theta == 0 or theta == 180:
            l, w, = l, w
        elif theta == 90 or theta == 270:
            l, w, = w, l

        x_min = x - l
        x_max = x + l
        y_min = y - w
        y_max = y + w

        furniture_df.loc[index, 'x_min'] = x_min
        furniture_df.loc[index, 'x_max'] = x_max
        furniture_df.loc[index, 'y_min'] = y_min
        furniture_df.loc[index, 'y_max'] = y_max

    return furniture_df


def get_function_feature(room_furnitureDf, out_wall_df, inner_df,  plot_function_area=False):

    function_name = list(room_furnitureDf['function'].drop_duplicates().values)
    
    min_max_xy_ = min_max_xy(inner_df)

    min_x_norm = min_max_xy_['min_x']
    min_y_norm = min_max_xy_['min_y']
    max_x_norm = min_max_xy_['max_x']
    max_y_norm = min_max_xy_['max_y']

    features= []

    for name_ in function_name:

        furniture_temp = room_furnitureDf[room_furnitureDf['function'] == name_]

        furniture_x_min = min(min(furniture_temp['x_min']) , min(furniture_temp['x_max']))
        furniture_y_min = min(min(furniture_temp['y_min']), min(furniture_temp['y_max']))
        furniture_x_max = max(max(furniture_temp['x_min']) , max(furniture_temp['x_max']))
        furniture_y_max = max(max(furniture_temp['y_min']) , max(furniture_temp['y_max']))

        if plot_function_area:
            plt.plot((furniture_x_min, furniture_x_min, furniture_x_max, furniture_x_max, furniture_x_min),
                     (furniture_y_min, furniture_y_max, furniture_y_max, furniture_y_min, furniture_y_min), 'b')

        lx = furniture_x_max - furniture_x_min
        ly = furniture_y_max - furniture_y_min

        df_temp = pd.DataFrame([[min_x_norm, min_y_norm, max_x_norm, max_y_norm],
                                [furniture_x_min, furniture_y_min, furniture_x_max,
                                 furniture_y_max]], columns=['minX', 'minY', 'maxX', 'maxY'])


        diff_name = df_temp.diff().abs().loc[1, :].idxmin()

        if diff_name == 'minX':
            # furniture_df.loc[:, 'function_theta'][furniture_df['function'] == name_] = 90
            function_theta = 90
            function_x = furniture_x_min
            function_y = furniture_y_max
        elif diff_name == 'minY':
            # furniture_df.loc[:, 'function_theta'][furniture_df['function'] == name_] = 0
            function_theta = 0
            function_x = furniture_x_min
            function_y = furniture_y_min
        elif diff_name == 'maxX':
            # furniture_df.loc[:, 'function_theta'][furniture_df['function'] == name_] = 270
            function_theta = 270
            function_x = furniture_x_max
            function_y = furniture_y_min
        elif diff_name == 'maxY':
            # furniture_df.loc[:, 'function_theta'][furniture_df['function'] == name_] = 180
            function_theta = 180
            function_x = furniture_x_max
            function_y = furniture_y_max


        leftX = function_x / (max_x_norm - min_x_norm)
        leftY = function_y / (max_y_norm - min_y_norm)
        theta = function_theta
        length = lx / (max_x_norm - min_x_norm)
        width = ly / (max_y_norm - min_y_norm)
        name = name_
        features.append([leftX, leftY, theta, length, width, name])

    F_feature_df = pd.DataFrame(features, columns=['leftX', 'leftY', 'theta', 'length', 'width', 'name'])
    return F_feature_df


def get_CNN_data(data, fileid):

    # 获取基本信息
    materialDf = get_material_info(data['DesignMaterialList'])

    data_ = xmltodict.parse(data['XML'])['root']

    out_wall_df = get_outer_wall(data_)

    inner_wallDf = get_inner_wall(data_)

    inner_wallDf['@WallID'] = inner_wallDf.index

    roomDf = get_room(data_)
    doorDf, room_id_dict = process_door(data_)

    furnituresDf = get_furniture(data_)
    furnituresDf = pd.merge(furnituresDf, materialDf, how='left', on='@MaterialID')  # 合并家具与名字
    furnituresDf = furnituresDf.dropna()


    # 设置要寻找的房间Id
    room_name = '主卧'
    room_id = -11  # 默认值
    for index, row in roomDf.iterrows():
        if row['@RoomName'] in ['主卧', '主卧房']:
            room_id = index
            break
    if room_id == -11:
        # logger = logging.getLogger('room error')
        logging.error('cannot find room')
        raise TypeError


    room_furnitureDf = get_room_furniture(furnituresDf, str(room_id))
    if room_furnitureDf is None:
        print("room_furnitureDf is None")
        raise TypeError
    room_furnitureDf = room_furnitureDf.dropna()

    inner_df = inner_wallDf[inner_wallDf['@RoomID'] == room_id]
    door_list = []

    for k, v in room_id_dict.items():
        if room_id in v:
            door_list.append(k)

    doorDf = doorDf[doorDf['doorID'].isin(door_list)]

    out_wall_df = process_out_wall(out_wall_df, inner_df)
    room_min_max = min_max_xy( inner_df)

    norm_x = lambda x: (x - room_min_max['min_x']) / (room_min_max['max_x'] - room_min_max['min_x'])
    norm_y = lambda y: (y - room_min_max['min_y']) / (room_min_max['max_y'] - room_min_max['min_y'])
    room_furnitureDf['x_min'] = room_furnitureDf['x_min'].apply(norm_x)
    room_furnitureDf['x_max'] = room_furnitureDf['x_max'].apply(norm_x)
    room_furnitureDf['y_min'] = room_furnitureDf['y_min'].apply(norm_y)
    room_furnitureDf['y_max'] = room_furnitureDf['y_max'].apply(norm_y)

    inner_df['@StartX'] = inner_df['@StartX'].apply(norm_x)
    inner_df['@EndX'] = inner_df['@EndX'].apply(norm_x)
    inner_df['@StartY'] = inner_df['@StartY'].apply(norm_y)
    inner_df['@EndY'] = inner_df['@EndY'].apply(norm_y)

    out_wall_df['@StartX'] = out_wall_df['@StartX'].apply(norm_x)
    out_wall_df['@EndX'] = out_wall_df['@EndX'].apply(norm_x)
    out_wall_df['@StartY'] = out_wall_df['@StartY'].apply(norm_y)
    out_wall_df['@EndY'] = out_wall_df['@EndY'].apply(norm_y)

    doorDf['@StartX'] = doorDf['@StartX'].apply(norm_x)
    doorDf['@EndX'] = doorDf['@EndX'].apply(norm_x)
    doorDf['@StartY'] = doorDf['@StartY'].apply(norm_y)
    doorDf['@EndY'] = doorDf['@EndY'].apply(norm_y)

    F_feature_df = get_function_feature(room_furnitureDf, out_wall_df, inner_df)


    door_df = doorDf[doorDf["@Type"]=="DOOR"]
    window_df = doorDf[doorDf["@Type"]=="Window"]
    for index, row in door_df.iterrows():
        plt.plot((row['@StartX'], row['@EndX']), (row['@StartY'], row['@EndY'],), 'g', linewidth=4.0)

    for index, row in window_df.iterrows():
        plt.plot((row['@StartX'], row['@EndX']), (row['@StartY'], row['@EndY'],), 'c', linewidth=4.0)
    # for index, row in out_wall_df.iterrows():
    #     plt.plot((row['@StartX'], row['@EndX']), (row['@StartY'], row['@EndY']), "r")
    for index, row in inner_df.iterrows():
        plt.plot((row['@StartX'], row['@EndX']), (row['@StartY'], row['@EndY']), 'r')

    # for index, row in room_furnitureDf.iterrows():
    #     plt.plot((row['x_min'], row['x_min'], row['x_max'], row['x_max'], row['x_min']), (row['y_min'], row['y_max'], row['y_max'], row['y_min'], row['y_min']))
    # plt.show()

    if F_feature_df.shape[0]>0:
        plt.axis('off')
        plt.savefig("./data/image/"+fileid+".png", dpi=10)
        plt.clf()


def iter_schemes(files):
    count = 0
    for __file in files[:3000]:
        try:
            with open(__file) as fh:
                json_data = json.load(fh)
                json_data =json.loads(json_data)
                count += 1
        except Exception as e:
            import traceback
            logging.critical('read error, {}, {}'.format(traceback.print_exc(), __file))
            pass
        yield json_data, __file


def save_res(filename):
    try:
        file_id = filename.split("/")[-1].replace(".json", "")
        data = json.load(open(filename, 'r'))
        data = json.loads(data)
        get_CNN_data(data, file_id)
    except Exception as e:
        print("{} json data can't parsed!".format(filename))


if __name__ == '__main__':
    scheme_dir = r"\\SWJ20180131005\xml2"
    from glob import glob
    files = glob(scheme_dir+"/*.json")
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    pool.map(save_res, files)
    pool.close()
    pool.join()


    # fname = r"\\SWJ20180131005\xml2\02341710.json"
    # data = json.load(open(fname, 'r'))
    # data = json.loads(data)
    # F_feature_df = get_CNN_data(data, '08276323')
    # print(F_feature_df)


