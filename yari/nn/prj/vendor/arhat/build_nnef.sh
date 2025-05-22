#!/bin/bash

# SPDX-FileCopyrightText: © 2025 FRAGATA COMPUTER SYSTEMS AG
#
# SPDX-License-Identifier: Apache-2.0

cd ../../../src/vendor/arhat/nnef
go build -o ../../../../bin/vendor/arhat/image_to_tensor -mod=mod fragata/arhat/nnef/tools/image_to_tensor


