fire train (2000): 0.774335416711
fire test (2000): 0.852171272227

head test/train:  LMEDS   1.0765431032088326
                  RANSAC  1.1547205416656348, 1.3650544841105936

office test/train 1.26, 1.25
            
heads train (1000): 1.38840533254
heads test (1000): 1.15472054167

redkitchen_Train_15.csv median 1.16451781794
stairs_Train_15.csv     median 1.51468257953
pumpkin_Train_15.csv    median 1.29073101975

heads
2000-40
iteration: 571 44.1006285263 279.19249173 235.091863204 3.64066121129 7.78289296222 time 0:00:10.468065
iteration: 4400 31.0958453349 287.966950323 256.871104988 2.75984065176 7.62594638681 time 0:03:47.351766

heads median 3881 2.592033134206067
processed  heads Test 400000 5 20 0:16:35.589945
stair 4312 5.51 (1.9)


  // Essential matrix.
  Eigen::Matrix3d E;
  // Fundamental matrix.
  Eigen::Matrix3d F;
  // Homography matrix.
  Eigen::Matrix3d H;
  
  output /home/weihao/Projects/tmp/fire_Test.csv /home/weihao/Projects/p_files/Test_fire.p
median 9240 2.9703977874609935
output /home/weihao/Projects/tmp/office_Test.csv /home/weihao/Projects/p_files/Test_office.p
median 12577 3.235353009095589

Rotations

/home/weihao/Projects/tmp/heads_Test.csv
total data 3881 out of 3881.
Total data 3881 3881
Truth (3,)
medians: (x,y,z,r)  1.3011721960403941 1.448570404587601 0.713444043607908 2.5920331341841045
values, fun [-0.0184538  -0.00132773 -0.00924446  1.        ] 14.645431711
[[ 0.99995639  0.00924432 -0.00132773]
 [-0.00921825  0.99978724  0.01845274]
 [ 0.00149803 -0.01843969  0.99982885]]
3136 0.04649509421745931 0.004670099397624599

/home/weihao/Projects/tmp/office_Test_0.csv
total data 2832 out of 2832.
Total data 2832 2832
Truth (3,)
medians: (x,y,z,r)  0.8214952019578357 1.5436046419416543 0.5913601513127889 2.3143972840069984
values, fun [-0.03627774 -0.00573483 -0.01591558  1.        ] 8.56041170525
[[ 0.99985691  0.01591464 -0.0057348 ]
 [-0.01569646  0.99921878  0.03626919]
 [ 0.00630753 -0.03617398  0.9993256 ]]
2527 0.03384406363483834 0.0033875788307278364

/home/weihao/Projects/tmp/office_Test_1.csv
total data 3034 out of 3034.
Total data 3034 3034
Truth (3,)
medians: (x,y,z,r)  0.8752525400164227 1.4200367432522996 0.5338025372493024 2.0708773015495656
values, fun [-0.01569969  0.0054762  -0.02734908  1.        ] 6.90706430349
[[ 0.99961105  0.02734526  0.00547617]
 [-0.02742824  0.99950049  0.01569881]
 [-0.00504415 -0.0158429   0.99986177]]
2748 0.025014268967543794 0.0025134877378067974
0:02:21.364233

/home/weihao/Projects/tmp/stairs_Test.csv
total data 4312 out of 4312.
Total data 4312 4312
Truth (3,)
medians: (x,y,z,r)  2.645984533624836 2.4142265402961858 1.3699297801677868 5.509610956704533
values, fun [-0.08804063  0.07521243 -0.01873242  1.        ] 92.4421143102
[[ 0.99699793  0.01867837  0.07514154]
 [-0.02526458  0.9958284   0.08767836]
 [-0.07319039 -0.08931356  0.99331075]]
3924 0.22999647648100338 0.02355813310658754

heads
rms, median 916.1802678803509 1.4052827032956887
match point 179.068 154.805
median 0.5434944112621545

median kitty_02 0.0908562941520348
median office 0.444759
median heads 0.523

-1.7771875178057654e-07 0.28229383413037923 0.23430977114962356heads_Test_cv_s3.p
-1.296129136484579e-07 0.23764431214806736 0.2218056141002258
-8.988266690417197e-08 0.23309543803325997 0.21475937398480888

-3.15e-08 0.203578 0.1950618

iteration: 1000 0.895871 1.803932 0.627861 1.098989 

[feature=20, (1024,128), 0.288 vs 0.678]

heads_Test_cv_1.p median heads 0.41379872862
heads_Test_cv_m5.p median heads 2.1223335316521887
office_Test_cv_s1 median office 0.3139776077104028

heads: 0.55, 0.22, 0.40, 0.11
heads_m5: 2.13, 0.98, 1.30, 0.54
heads_L: 0.51, 0.18, 0.32, 0.11
Kitti: 0.35 0.14 0.14 0.12

heads_Test_cv_s3.p 0.523

median pump_Train_cv_s3.p 0.3991440449500680 0.963315155119572 0.14819964728586943 0.19164183564372456 0.14400667655616972
median redk_Train_cv_s3.p 0.3694888721644974 0.708563031149695 0.13393579593127025 0.21726652308925098 0.1136098640799859
median fire_Train_cv_s3.p 0.3565516961681562 0.863721171648828 0.15087688034469604 0.21755542473743428 0.09359544164935729
median ches_Train_cv_s3.p 0.3480354912722443 0.931744727934004 0.12985054137956903 0.22947614023600604 0.10029237300355022
median offi_Train_cv_s3.p 0.4040077353166283 1.218636887586839 0.1410810468445025  0.27579824119606844 0.10760890731302941
median head_Train_cv_s3.p 0.582598891625068  1.033552425387086 0.2994505848271997  0.3339139021846177  0.14001323564016555
median head_Test_cv_s3.p  0.5196997469881491 1.220140825985599 0.20826865904063163 0.3243356105709303  0.12374250408063947

median Train_heads_cm_m20.p 4.182476647599993 22.23802908393342 1.903963434684595 1.997724808195505 0.9862274740241368
median Test_heads_cm_m20.p  3.642896210890033 22.00508609612644 1.882101458868611 1.653697391548522 1.1009306953531741

median heads_Test_cv.p, 0.5435 1.1787 0.2141 0.3614 0.1198
median kitty_02_Test_cv.p,  0.0910 0.6203 0.0310 0.0364 0.0505
median kitty_00_Train_cv.p, 0.0959 0.6672 0.0308 0.0391 0.0472

kitty fc_20
iteration: 4.59 9.268 1.54 1.977 3.735
