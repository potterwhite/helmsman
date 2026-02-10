/*
 * Copyright (c) 2025 PotterWhite
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include "Network/pch.h"

namespace arcforge {
namespace embedded {
namespace network_socket {

extern const std::string_view kcurrent_lib_name;

class Base;

enum class SocketStatus { kvalid = 0x31, kinvalid = 0x32, kunknowerror = 0x3f };
enum class SocketReturnValue {
	ksuccess = 0x49,
	// ***************************
	// --- recv opts errors ---
	kreceived_null = 0x50,
	kreceived_illegal = 0x51,
	kreceivelength_failed = 0x52,
	// --- send opts errors ---
	ksendcount_failed = 0x60,
	ksenddata_failed = 0x61,
	ksendlength_failed = 0x62,
	// --- configuration errors ---
	kcount_too_large = 0x70,
	kempty_string = 0x71,
	kfd_illegal = 0x72,
	keof = 0x73,
	ksocketpath_empty = 0x74,
	// --- posix api errors ---
	kconnect_server_failed = 0x80,
	klisten_error = 0x81,
	kbind_error = 0x82,
	kpeer_abnormally_closed = 0x83,
	kaccept_timeout = 0x84,
	ksetsocketopt_error = 0x85,
	// --- impl layer errors ---
	kimpl_nullptr_error = 0x90,
	// ***************************
	kinit_state = 0xff,
	kunknownerror = 0x100
};

std::string SocketReturnValueToString(SocketReturnValue value);
bool SocketReturnValueIsSuccess(SocketReturnValue value);

struct SocketAcceptReturn {
	SocketReturnValue return_value;
	std::unique_ptr<Base> client;

	SocketAcceptReturn();

	SocketAcceptReturn(SocketReturnValue value, std::unique_ptr<Base> client_ptr);
};

}  // namespace network_socket
}  // namespace embedded
}  // namespace arcforge