\ Model TSP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  3.6 x_0_1 + 5.1 x_0_2 + 10 x_0_3 + 15.3 x_0_4 + 20 x_0_5 + 16 x_0_6
   + 14.2 x_0_7 + 23 x_0_8 + 26.4 x_0_9 + 3.6 x_1_0 + 3.6 x_1_2 + 6.4 x_1_3
   + 12.1 x_1_4 + 18.1 x_1_5 + 13.2 x_1_6 + 10.6 x_1_7 + 19.7 x_1_8
   + 23 x_1_9 + 5.1 x_2_0 + 3.6 x_2_1 + 7.1 x_2_3 + 10.6 x_2_4 + 15 x_2_5
   + 15.8 x_2_6 + 10.8 x_2_7 + 18.4 x_2_8 + 21.9 x_2_9 + 10 x_3_0
   + 6.4 x_3_1 + 7.1 x_3_2 + 7 x_3_4 + 15.7 x_3_5 + 10 x_3_6 + 4.2 x_3_7
   + 13.9 x_3_8 + 17 x_3_9 + 15.3 x_4_0 + 12.1 x_4_1 + 10.6 x_4_2 + 7 x_4_3
   + 9.9 x_4_5 + 15.3 x_4_6 + 5 x_4_7 + 7.8 x_4_8 + 11.3 x_4_9 + 20 x_5_0
   + 18.1 x_5_1 + 15 x_5_2 + 15.7 x_5_3 + 9.9 x_5_4 + 25 x_5_6 + 14.9 x_5_7
   + 12 x_5_8 + 15 x_5_9 + 16 x_6_0 + 13.2 x_6_1 + 15.8 x_6_2 + 10 x_6_3
   + 15.3 x_6_4 + 25 x_6_5 + 10.3 x_6_7 + 19.2 x_6_8 + 21 x_6_9
   + 14.2 x_7_0 + 10.6 x_7_1 + 10.8 x_7_2 + 4.2 x_7_3 + 5 x_7_4
   + 14.9 x_7_5 + 10.3 x_7_6 + 10.2 x_7_8 + 13 x_7_9 + 23 x_8_0
   + 19.7 x_8_1 + 18.4 x_8_2 + 13.9 x_8_3 + 7.8 x_8_4 + 12 x_8_5
   + 19.2 x_8_6 + 10.2 x_8_7 + 3.6 x_8_9 + 26.4 x_9_0 + 23 x_9_1
   + 21.9 x_9_2 + 17 x_9_3 + 11.3 x_9_4 + 15 x_9_5 + 21 x_9_6 + 13 x_9_7
   + 3.6 x_9_8
Subject To
 out0: x_0_1 + x_0_2 + x_0_3 + x_0_4 + x_0_5 + x_0_6 + x_0_7 + x_0_8
   + x_0_9 = 1
 in0: x_1_0 + x_2_0 + x_3_0 + x_4_0 + x_5_0 + x_6_0 + x_7_0 + x_8_0 + x_9_0
   = 1
 out1: x_1_0 + x_1_2 + x_1_3 + x_1_4 + x_1_5 + x_1_6 + x_1_7 + x_1_8
   + x_1_9 = 1
 in1: x_0_1 + x_2_1 + x_3_1 + x_4_1 + x_5_1 + x_6_1 + x_7_1 + x_8_1 + x_9_1
   = 1
 out2: x_2_0 + x_2_1 + x_2_3 + x_2_4 + x_2_5 + x_2_6 + x_2_7 + x_2_8
   + x_2_9 = 1
 in2: x_0_2 + x_1_2 + x_3_2 + x_4_2 + x_5_2 + x_6_2 + x_7_2 + x_8_2 + x_9_2
   = 1
 out3: x_3_0 + x_3_1 + x_3_2 + x_3_4 + x_3_5 + x_3_6 + x_3_7 + x_3_8
   + x_3_9 = 1
 in3: x_0_3 + x_1_3 + x_2_3 + x_4_3 + x_5_3 + x_6_3 + x_7_3 + x_8_3 + x_9_3
   = 1
 out4: x_4_0 + x_4_1 + x_4_2 + x_4_3 + x_4_5 + x_4_6 + x_4_7 + x_4_8
   + x_4_9 = 1
 in4: x_0_4 + x_1_4 + x_2_4 + x_3_4 + x_5_4 + x_6_4 + x_7_4 + x_8_4 + x_9_4
   = 1
 out5: x_5_0 + x_5_1 + x_5_2 + x_5_3 + x_5_4 + x_5_6 + x_5_7 + x_5_8
   + x_5_9 = 1
 in5: x_0_5 + x_1_5 + x_2_5 + x_3_5 + x_4_5 + x_6_5 + x_7_5 + x_8_5 + x_9_5
   = 1
 out6: x_6_0 + x_6_1 + x_6_2 + x_6_3 + x_6_4 + x_6_5 + x_6_7 + x_6_8
   + x_6_9 = 1
 in6: x_0_6 + x_1_6 + x_2_6 + x_3_6 + x_4_6 + x_5_6 + x_7_6 + x_8_6 + x_9_6
   = 1
 out7: x_7_0 + x_7_1 + x_7_2 + x_7_3 + x_7_4 + x_7_5 + x_7_6 + x_7_8
   + x_7_9 = 1
 in7: x_0_7 + x_1_7 + x_2_7 + x_3_7 + x_4_7 + x_5_7 + x_6_7 + x_8_7 + x_9_7
   = 1
 out8: x_8_0 + x_8_1 + x_8_2 + x_8_3 + x_8_4 + x_8_5 + x_8_6 + x_8_7
   + x_8_9 = 1
 in8: x_0_8 + x_1_8 + x_2_8 + x_3_8 + x_4_8 + x_5_8 + x_6_8 + x_7_8 + x_9_8
   = 1
 out9: x_9_0 + x_9_1 + x_9_2 + x_9_3 + x_9_4 + x_9_5 + x_9_6 + x_9_7
   + x_9_8 = 1
 in9: x_0_9 + x_1_9 + x_2_9 + x_3_9 + x_4_9 + x_5_9 + x_6_9 + x_7_9 + x_8_9
   = 1
 MTZ01: 11 x_0_1 + u0 - u1 <= 10
 MTZ02: 11 x_0_2 + u0 - u2 <= 10
 MTZ03: 11 x_0_3 + u0 - u3 <= 10
 MTZ04: 11 x_0_4 + u0 - u4 <= 10
 MTZ05: 11 x_0_5 + u0 - u5 <= 10
 MTZ06: 11 x_0_6 + u0 - u6 <= 10
 MTZ07: 11 x_0_7 + u0 - u7 <= 10
 MTZ08: 11 x_0_8 + u0 - u8 <= 10
 MTZ09: 11 x_0_9 + u0 - u9 <= 10
 MTZ10: 11 x_1_0 - u0 + u1 <= 10
 MTZ12: 11 x_1_2 + u1 - u2 <= 10
 MTZ13: 11 x_1_3 + u1 - u3 <= 10
 MTZ14: 11 x_1_4 + u1 - u4 <= 10
 MTZ15: 11 x_1_5 + u1 - u5 <= 10
 MTZ16: 11 x_1_6 + u1 - u6 <= 10
 MTZ17: 11 x_1_7 + u1 - u7 <= 10
 MTZ18: 11 x_1_8 + u1 - u8 <= 10
 MTZ19: 11 x_1_9 + u1 - u9 <= 10
 MTZ20: 11 x_2_0 - u0 + u2 <= 10
 MTZ21: 11 x_2_1 - u1 + u2 <= 10
 MTZ23: 11 x_2_3 + u2 - u3 <= 10
 MTZ24: 11 x_2_4 + u2 - u4 <= 10
 MTZ25: 11 x_2_5 + u2 - u5 <= 10
 MTZ26: 11 x_2_6 + u2 - u6 <= 10
 MTZ27: 11 x_2_7 + u2 - u7 <= 10
 MTZ28: 11 x_2_8 + u2 - u8 <= 10
 MTZ29: 11 x_2_9 + u2 - u9 <= 10
 MTZ30: 11 x_3_0 - u0 + u3 <= 10
 MTZ31: 11 x_3_1 - u1 + u3 <= 10
 MTZ32: 11 x_3_2 - u2 + u3 <= 10
 MTZ34: 11 x_3_4 + u3 - u4 <= 10
 MTZ35: 11 x_3_5 + u3 - u5 <= 10
 MTZ36: 11 x_3_6 + u3 - u6 <= 10
 MTZ37: 11 x_3_7 + u3 - u7 <= 10
 MTZ38: 11 x_3_8 + u3 - u8 <= 10
 MTZ39: 11 x_3_9 + u3 - u9 <= 10
 MTZ40: 11 x_4_0 - u0 + u4 <= 10
 MTZ41: 11 x_4_1 - u1 + u4 <= 10
 MTZ42: 11 x_4_2 - u2 + u4 <= 10
 MTZ43: 11 x_4_3 - u3 + u4 <= 10
 MTZ45: 11 x_4_5 + u4 - u5 <= 10
 MTZ46: 11 x_4_6 + u4 - u6 <= 10
 MTZ47: 11 x_4_7 + u4 - u7 <= 10
 MTZ48: 11 x_4_8 + u4 - u8 <= 10
 MTZ49: 11 x_4_9 + u4 - u9 <= 10
 MTZ50: 11 x_5_0 - u0 + u5 <= 10
 MTZ51: 11 x_5_1 - u1 + u5 <= 10
 MTZ52: 11 x_5_2 - u2 + u5 <= 10
 MTZ53: 11 x_5_3 - u3 + u5 <= 10
 MTZ54: 11 x_5_4 - u4 + u5 <= 10
 MTZ56: 11 x_5_6 + u5 - u6 <= 10
 MTZ57: 11 x_5_7 + u5 - u7 <= 10
 MTZ58: 11 x_5_8 + u5 - u8 <= 10
 MTZ59: 11 x_5_9 + u5 - u9 <= 10
 MTZ60: 11 x_6_0 - u0 + u6 <= 10
 MTZ61: 11 x_6_1 - u1 + u6 <= 10
 MTZ62: 11 x_6_2 - u2 + u6 <= 10
 MTZ63: 11 x_6_3 - u3 + u6 <= 10
 MTZ64: 11 x_6_4 - u4 + u6 <= 10
 MTZ65: 11 x_6_5 - u5 + u6 <= 10
 MTZ67: 11 x_6_7 + u6 - u7 <= 10
 MTZ68: 11 x_6_8 + u6 - u8 <= 10
 MTZ69: 11 x_6_9 + u6 - u9 <= 10
 MTZ70: 11 x_7_0 - u0 + u7 <= 10
 MTZ71: 11 x_7_1 - u1 + u7 <= 10
 MTZ72: 11 x_7_2 - u2 + u7 <= 10
 MTZ73: 11 x_7_3 - u3 + u7 <= 10
 MTZ74: 11 x_7_4 - u4 + u7 <= 10
 MTZ75: 11 x_7_5 - u5 + u7 <= 10
 MTZ76: 11 x_7_6 - u6 + u7 <= 10
 MTZ78: 11 x_7_8 + u7 - u8 <= 10
 MTZ79: 11 x_7_9 + u7 - u9 <= 10
 MTZ80: 11 x_8_0 - u0 + u8 <= 10
 MTZ81: 11 x_8_1 - u1 + u8 <= 10
 MTZ82: 11 x_8_2 - u2 + u8 <= 10
 MTZ83: 11 x_8_3 - u3 + u8 <= 10
 MTZ84: 11 x_8_4 - u4 + u8 <= 10
 MTZ85: 11 x_8_5 - u5 + u8 <= 10
 MTZ86: 11 x_8_6 - u6 + u8 <= 10
 MTZ87: 11 x_8_7 - u7 + u8 <= 10
 MTZ89: 11 x_8_9 + u8 - u9 <= 10
 MTZ90: 11 x_9_0 - u0 + u9 <= 10
 MTZ91: 11 x_9_1 - u1 + u9 <= 10
 MTZ92: 11 x_9_2 - u2 + u9 <= 10
 MTZ93: 11 x_9_3 - u3 + u9 <= 10
 MTZ94: 11 x_9_4 - u4 + u9 <= 10
 MTZ95: 11 x_9_5 - u5 + u9 <= 10
 MTZ96: 11 x_9_6 - u6 + u9 <= 10
 MTZ97: 11 x_9_7 - u7 + u9 <= 10
 MTZ98: 11 x_9_8 - u8 + u9 <= 10
Bounds
Binaries
 x_0_1 x_0_2 x_0_3 x_0_4 x_0_5 x_0_6 x_0_7 x_0_8 x_0_9 x_1_0 x_1_2 x_1_3
 x_1_4 x_1_5 x_1_6 x_1_7 x_1_8 x_1_9 x_2_0 x_2_1 x_2_3 x_2_4 x_2_5 x_2_6
 x_2_7 x_2_8 x_2_9 x_3_0 x_3_1 x_3_2 x_3_4 x_3_5 x_3_6 x_3_7 x_3_8 x_3_9
 x_4_0 x_4_1 x_4_2 x_4_3 x_4_5 x_4_6 x_4_7 x_4_8 x_4_9 x_5_0 x_5_1 x_5_2
 x_5_3 x_5_4 x_5_6 x_5_7 x_5_8 x_5_9 x_6_0 x_6_1 x_6_2 x_6_3 x_6_4 x_6_5
 x_6_7 x_6_8 x_6_9 x_7_0 x_7_1 x_7_2 x_7_3 x_7_4 x_7_5 x_7_6 x_7_8 x_7_9
 x_8_0 x_8_1 x_8_2 x_8_3 x_8_4 x_8_5 x_8_6 x_8_7 x_8_9 x_9_0 x_9_1 x_9_2
 x_9_3 x_9_4 x_9_5 x_9_6 x_9_7 x_9_8
End
