#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 08:31:33 2017

@author: david
"""
a = [-0.35749488,  1.75846923,  0.14105786, -0.56960297,  0.32528622,
        0.39986652,  1.39383682,  0.33094672, -1.37787941,  0.30667804,
       -1.61454981,  1.47823979,  1.65220608,  0.5214557 ,  0.42874704,
       -1.12052743, -0.55925472, -1.81167633,  0.16579398, -0.95699126,
       -1.0396402 ,  0.09677255,  0.3731544 ,  0.95831226, -0.55330313,
       -0.41136056,  0.4234974 , -0.35858288, -0.84245861, -0.01826373,
        0.58588503, -2.2642798 , -1.1130473 , -0.37691512,  1.23575884,
       -0.70563671,  1.18733023,  1.40843057,  1.38872578,  0.06605212,
       -1.33491955,  1.00809178, -1.4253874 ,  1.03812981,  0.5750469 ,
       -2.00296875, -0.09376887, -0.7295192 , -0.31227703,  0.11553579,
        0.33130782,  0.17777402, -0.51015014, -0.49234234, -1.17178339,
       -0.17965018, -1.20949754,  0.88462814, -1.27770125, -0.50778386,
        2.56884483, -0.16381047, -0.33559308,  0.36066625, -0.79019568,
        1.30799167,  0.41654739,  1.32948749,  0.30170328,  1.74168779,
       -1.92891668,  0.67494497, -1.92579113, -1.26452614, -1.30048632,
       -0.4349639 ,  0.91270418,  0.87871163,  1.47876792, -0.21360812,
        2.24883211,  0.53595993,  1.92618164, -3.12711789,  2.58701089,
        1.18325926,  1.37762413, -1.08271724, -0.11673398,  0.740243  ,
        1.943081  , -0.81486057, -2.24167039,  0.05744477, -0.38011589,
        1.07614803,  1.14564895, -0.33931181,  1.01730095,  0.88247665]

b = [-0.37187   ,  1.75405607,  0.15164789, -0.57074843,  0.32522189,
        0.39687672,  1.38194433,  0.34511395, -1.38319105,  0.30524606,
       -1.60211967,  1.46952247,  1.6242529 ,  0.51639162,  0.42648656,
       -1.12525354, -0.54203139, -1.79779877,  0.16336382, -0.94944767,
       -1.0350904 ,  0.09972839,  0.38441884,  0.95252474, -0.54931266,
       -0.40748373,  0.40695836, -0.33720464, -0.83768921, -0.01768967,
        0.59681448, -2.25529212, -1.0929347 , -0.36899958,  1.24013572,
       -0.69619358,  1.18392058,  1.40941187,  1.37743854,  0.07139806,
       -1.34393532,  0.99289025, -1.42990542,  1.03944425,  0.58185034,
       -1.99521692, -0.08933417, -0.7267275 , -0.30252598,  0.10938809,
        0.33584539,  0.1787626 , -0.52446215, -0.50010934, -1.17085213,
       -0.18167088, -1.21755143,  0.89218122, -1.27055179, -0.50533409,
        2.56556848, -0.14835272, -0.35355257,  0.36603672, -0.77594328,
        1.31304653,  0.42938606,  1.32675645,  0.2750862 ,  1.72612495,
       -1.92693023,  0.67921847, -1.90845633, -1.26924071, -1.31433073,
       -0.43676274,  0.90945337,  0.87868702,  1.47962907, -0.2176123 ,
        2.24330696,  0.53178577,  1.92278791, -3.12547033,  2.58735518,
        1.18709384,  1.38184458, -1.08586105, -0.1052644 ,  0.73881069,
        1.93419484, -0.80188212, -2.24605363,  0.05025934, -0.38625702,
        1.10065159,  1.14833475, -0.33663625,  1.03419113,  0.8758707 ]