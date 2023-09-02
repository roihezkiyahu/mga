box_classes = ["plot", "x_tick", "y_tick", "scatter_point", "bar", "dot_point", "line_point", "tick_label"]
colors_list = ["r", "blue", "blue", "black", "purple", "yellow", "green", "orange"]
chart_labels = ["line", "scatter", "vertical_bar", "horizontal_bar", "dot"]

chart_labels_2_indx = {class_name: idx for idx, class_name in enumerate(chart_labels)}
indx_2_chart_label = {idx: class_name for idx, class_name in enumerate(chart_labels)}

class_box_to_idx = {class_name: idx for idx, class_name in enumerate(box_classes)}
idx_to_class_box = {idx: class_name for idx, class_name in enumerate(box_classes)}

outlier_images = ['9a6b0f8d2add', '31bf91c2135f', '5c8a334d4c2a', 'd5545cc232eb', '5d64fd0ed8e0', '63728f0ac9d4',
                  '7ec413758225', '1b005c83b69f', '364b84297ac5',
                  '3131ea2138d9', '7fcff2982cc5', 'e85e032b0769', '4a4069e10290', '140ac1bc7993', '2cad54f759d8',
                  '908b4f8dd53f', '28fd2dca5548', '410fe5700912', '5c41897b0b4f', '21469c559685', '0f2d2f3fa0c7',
                  '31342d25fd04', '6a58889a856c', '06892a320354', '76a1114020df', '4710ac412610', '63750a64c3ed',
                  '83e353878205', '5aceb4acb123', '782552762737', '2b3d51319745', '86ff705e7b95', 'f213ab5cecab',
                  'e3378f05270f', '41f10d9bba46', '07efcd88ec60', '54fe5edf517b', '5db0c2fbffbf', '0e46519b2c18',
                  '87aa017636d1', 'e5d467c0480c', '17ddd52fad22', '72711f84a925',
                  'fe82e254f88f', '22064416c687', '4136ee97c355', '8d520bfed188', 'dad5d9895f2c',
                  '87914f25a0d8', '399807d34c50', '567cb02f6d77', '40691ebcc4af', 'a1f6d29fbf03', '6d4366d7a6d4',
                  '44895ab2b62c', 'f36c5b0b6892', 'c0e9ace459da', '0813b92962da',
                  '2d3232e2dafb', '9c49cc71ce8c', 'ae4db983eacc', '296b04ffa37e',
                  'd894e055c0e5', '856f223a6a7f', '9c17d5ccb2bf', '6d2a47a759bd', '29f62e74bd03', '849d1309ab73',
                  '3ea4d282dbf9', '2a9bae9e31b0', '6adfc00a2d50', 'e9d97c38a655',
                  '81f2be455798', 'eb3c564a48c9', '22fea63c7cd4', 'fe17016c1954', '44da118ec281', '80eba770dd5c',
                  '44dcd7d07d32', '66c37140aeb0', '99221d50d828', '2c3097ab504b', '20dab9b77cc9',
                  '0e9fb4d74db6', '5bf5e9c11a5a', '2bc5e5112227', '7872db0dce00', '76b62aacde66',
                  '93e48d8fd933', '67b435ea3c0e', '0303caad407a', 'a884bc53ccfe', '419f29e57bbe', '81cc448ac5e0',
                  '1ff7c08c0c99',
                  '1ff2a3d9f346', 'ffd07dd00bea', '1b01876cc4ea', '3ba7eea1b54e', '1bd8fd1dac82', '27bab191dca3',
                  '80cec12b04d5', 'bc3491808b70', 'dbd30a009589', '4bb2172ccbb3', '16f6e43e3d58', 'c4a88591e6e7',
                  'f410a9345d4e', 'f73d744179f3', 'ef428c966680', 'b5212d9d9c3b', '054cdbc03461', '2d29837b0c5c',
                  '640c5465a9d7', '42e90b337866', '9ac926ee5687',
                  '62adada1f463', '9e905c59d6f4', '9e318087cdc7', '5c02061e5289', 'a16e2ec7e3a9', 'f9a7c7ce5af3',
                  'c106936ab6ff', '19516f0cbba7', '2e66951568b1', '05890487983c', '75ec70a7fd39', 'f8ae787e0900',
                  '39d93547547d', 'add6cd8e1c4d', 'de90e79c1fa0',
                  '9c9135bbc350', '018722198a11', '1c1783d26408', '1d1278234de7', '4574f32062f2', '570f539d34f6',
                  'c9c33319024a', '56efa8b6ae39',
                  '32728f4a71ba', '0c81674b211a', '6d0a275e8d91', 'd5f8875369ef', 'b65f7c972138', '5ed8b796cfe8',
                  '634fcb429e61', '3884b5a15cd1',
                  '70a9e99c485b', "5b83424ed87c", "6c2e306f75dd", "2f1279289d66", "978eaf14b943", "504c5f47aa48",
                  "8f77cbb0abb3",
                  "94e410815a37", "293079f2c64a", "7350bf629093", "6e059f224f95", "9160b73291bf", "36d1f5d2ffff",
                  "9ce0972f5200", "173ac67e0806", "3fd8c5576ca6", "ee3ebe3d50ed", "17146ffecc38",
                  "1394c6271d7c", "e6d2dfc41e35", "4313ef4214ac", "0a472e31fb2c", "0c837d1f3260",
                  "b4649b22799f", "1465d45735fa", "f381236a998d", "b705588e47e9", "49c9e32a2cd1",
                  "3cf35e028270", "4d05f3bbccd9", "3f6268623aee", "a6ceda4f5521", "6e546ab6986c",
                  "e1bb13a43ccf", "f547467d453d", "e9303967ba74", "10928f154f28", "91e91cc6a0e7",
                  "07c31a091c16", "1215af1fbb4a", "8b865bd8f97e", "f4ca07ab539a", "3961d4816dd2",
                  "315218dc407c", "ba0d2e20eda2", "af9ad3de2561", "6aebf72022bb", "e2a66a887c75", "aaff10482865",
                  "03422fd0b855", "0bcd3bc4d26b", "da11087a4be7", "feb3842c7a00", "f1b5de9b43f", "a8d3fca15bf4",
                  "d0fc5fe01fbc", "d425749760fb", "8e7639628073", "f1b5d3e9b43f"]


