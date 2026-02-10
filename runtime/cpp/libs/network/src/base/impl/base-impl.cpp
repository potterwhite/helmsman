// Copyright (c) 2025 PotterWhite
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// libs/network/src/base/impl/base-impl.cpp
#include "Network/base/impl/base-impl.h"
#include "Utils/logger/logger.h"

namespace arcforge {
namespace embedded {
namespace network_socket {

// #define DEBUG
/*===================================================
 * constructors and operators
 *===================================================*/
BaseImpl::BaseImpl()
    : socketfd_(-1),
      socket_mutex_(std::make_unique<std::mutex>()),
      log_mutex_(std::make_unique<std::mutex>()) {
	arcforge::embedded::utils::Logger::GetInstance().Info("BaseImpl object constructed.",
	                                                      kcurrent_lib_name);
}

BaseImpl::~BaseImpl() {
	// socketfd_ = -1;
	closeSocket_safe();
	arcforge::embedded::utils::Logger::GetInstance().Info("BaseImpl cleaned up.",
	                                                      kcurrent_lib_name);
}

/*===================================================
 * getters & setters
 *===================================================*/
SocketStatus BaseImpl::isSocketFDValid_safe() {
	std::lock_guard<std::mutex> lock(*(socket_mutex_.get()));

	return isSocketFDValid();
}

SocketStatus BaseImpl::isSocketFDValid() const {

	if (socketfd_ > 0) {
		return SocketStatus::kvalid;
	} else {
		return SocketStatus::kinvalid;
	}
}

void BaseImpl::closeSocket_safe() {
	std::lock_guard<std::mutex> lock(*(socket_mutex_.get()));

	closeSocket();
}

void BaseImpl::closeSocket() {
	if (isSocketFDValid() == SocketStatus::kvalid) {
		close(socketfd_);
		socketfd_ = killegal_fd_value;
	} else {
		// log_error("socketfd is invalid, alert!!");
		arcforge::embedded::utils::Logger::GetInstance().Error("socketfd is invalid, alert!!",
		                                                       kcurrent_lib_name);
	}
}

int BaseImpl::getFD_safe() {
	std::lock_guard<std::mutex> lock(*(socket_mutex_.get()));

	return getFD();
}

int BaseImpl::getFD() const {
	return socketfd_;
}

void BaseImpl::setFD_safe(int fd) {
	std::lock_guard<std::mutex> lock(*(socket_mutex_.get()));

	setFD(fd);
}

void BaseImpl::setFD(int fd) {

	// socketfd_ = fd;
	//-----------------------------------------------
	// if we have a valid existing fd, close it first
	if (socketfd_ >= 0) {

		arcforge::embedded::utils::Logger::GetInstance().Info(
		    "BaseImpl::setFD - Closing existing socket FD: " + std::to_string(socketfd_),
		    kcurrent_lib_name);
		closeSocket();
	}
	socketfd_ = fd;
	if (fd >= 0) {

		arcforge::embedded::utils::Logger::GetInstance().Info(
		    "BaseImpl::setFD - New socket FD set: " + std::to_string(socketfd_), kcurrent_lib_name);
	} else {
		arcforge::embedded::utils::Logger::GetInstance().Info(
		    "BaseImpl::setFD - FD set to invalid value: " + std::to_string(fd), kcurrent_lib_name);
	}
}

const std::string& BaseImpl::getSocketPath_safe() {
	std::lock_guard<std::mutex> lock(*(socket_mutex_.get()));

	return getSocketPath();
}

const std::string& BaseImpl::getSocketPath() const {
	if (socketpath_.empty() == true) {
		// socket path verification
		// std::cerr << "LibArcForge_Network: Socket path is empty!" << std::endl;
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "LibArcForge_Network: Socket path is empty!", kcurrent_lib_name);
	}

	return socketpath_;
}

void BaseImpl::setSocketPath_safe(const std::string& path) {
	std::lock_guard<std::mutex> lock(*(socket_mutex_.get()));

	setSocketPath(path);
}

void BaseImpl::setSocketPath(const std::string& path) {

	socketpath_ = path;
}

// --- sendFloat_safe ---
SocketReturnValue BaseImpl::sendFloat_safe(const std::vector<float>& data) {
	std::lock_guard<std::mutex> lock(*(socket_mutex_.get()));

	if (socketfd_ < 0) {
		return SocketReturnValue::kfd_illegal;
	}

	uint32_t count = static_cast<uint32_t>(data.size());

	// transmit the length header
	if (::send(socketfd_, &count, sizeof(count), 0) != sizeof(count)) {
		// arcforge::embedded::utils::Logger::GetInstance().Info("sendFloat_safe: Failed to send count. errno: " + std::to_string(errno));
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "sendFloat_safe: Failed to send count. errno: " + std::to_string(errno),
		    kcurrent_lib_name);
		return SocketReturnValue::ksendcount_failed;
	}

	// transmit the data body (using a loop)
	if (count > 0) {
		size_t bytes_to_send = count * sizeof(float);
		size_t bytes_has_sent = 0;
		const char* data_ptr = reinterpret_cast<const char*>(data.data());

		while (bytes_has_sent < bytes_to_send) {
			ssize_t n_sent =
			    ::send(socketfd_, data_ptr + bytes_has_sent, bytes_to_send - bytes_has_sent, 0);
			if (n_sent < 0) {
				// arcforge::embedded::utils::Logger::GetInstance().Info("sendFloat_safe: send() error while sending data. errno: " +
				//     std::to_string(errno));
				arcforge::embedded::utils::Logger::GetInstance().Info(
				    "sendFloat_safe: send() error while sending data. errno: " +
				        std::to_string(errno),
				    kcurrent_lib_name);
				return SocketReturnValue::ksenddata_failed;
			}
			bytes_has_sent += static_cast<size_t>(n_sent);
		}
	}
	// arcforge::embedded::utils::Logger::GetInstance().Info("sendFloat_safe: Sent " + std::to_string(count) + " floats.");
	arcforge::embedded::utils::Logger::GetInstance().Info(
	    "sendFloat_safe: Sent " + std::to_string(count) + " floats.", kcurrent_lib_name);
	return SocketReturnValue::ksuccess;
}

// --- receiveFloat_safe  ---
SocketReturnValue BaseImpl::receiveFloat_safe(std::vector<float>& data) {
	std::lock_guard<std::mutex> lock(*(socket_mutex_.get()));

	// 1. fd validation verification
	if (socketfd_ < 0) {
		return SocketReturnValue::kfd_illegal;
	}

	// 2. clean all output container
	data.clear();

	// arcforge::embedded::utils::Logger::GetInstance().Info("receiveFloat_safe(): before ::recv line 113");
	arcforge::embedded::utils::Logger::GetInstance().Debug(
	    "receiveFloat_safe(): before ::recv line 113", kcurrent_lib_name);
	// 3. receive length that we need to read the data
	uint32_t count;
	ssize_t first_recv = ::recv(socketfd_, &count, sizeof(count), 0);
	// arcforge::embedded::utils::Logger::GetInstance().Info("receiveFloat_safe(): after ::recv line 117");
	arcforge::embedded::utils::Logger::GetInstance().Debug(
	    "receiveFloat_safe(): after ::recv line 117", kcurrent_lib_name);
	if (first_recv == 0) {
		// means received nothing
		// arcforge::embedded::utils::Logger::GetInstance().Info("receiveFloat_safe: Peer closed connection while trying to receive count.");
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "receiveFloat_safe: Peer closed connection while trying to receive count.",
		    kcurrent_lib_name);

		return SocketReturnValue::kpeer_abnormally_closed;
	}
	if (first_recv < 0) {
		// means something wrong when sending data
		// arcforge::embedded::utils::Logger::GetInstance().Info("receiveFloat_safe: recv() error while receiving count. errno: " +
		//     std::to_string(errno) + " (" + strerror(errno) + ")");
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "receiveFloat_safe: recv() error while receiving count. errno: " +
		        std::to_string(errno) + " (" + strerror(errno) + ")",
		    kcurrent_lib_name);
		return SocketReturnValue::kreceived_illegal;
	}
	if (first_recv != sizeof(count)) {
		// read wrong length of uint32_t, so error occurs
		// arcforge::embedded::utils::Logger::GetInstance().Info("receiveFloat_safe: Failed to receive count ");
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "receiveFloat_safe: Failed to receive count ", kcurrent_lib_name);
		return SocketReturnValue::kreceivelength_failed;
	}

	std::ostringstream temp_str;
	temp_str << "receiveFloat_safe: count=" << count;
	// arcforge::embedded::utils::Logger::GetInstance().Info(temp_str.str());
	arcforge::embedded::utils::Logger::GetInstance().Info(temp_str.str(), kcurrent_lib_name);

	// 4. optional: sanity check on count
	const uint32_t MAX_ALLOWED_FLOATS = 1024 * 1024;
	if (count > MAX_ALLOWED_FLOATS) {
		// arcforge::embedded::utils::Logger::GetInstance().Info("receiveFloat_safe: Received count (" + std::to_string(count) +
		//     ") exceeds MAX_ALLOWED_FLOATS (" + std::to_string(MAX_ALLOWED_FLOATS) + "). Aborting.");
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "receiveFloat_safe: Received count (" + std::to_string(count) +
		        ") exceeds MAX_ALLOWED_FLOATS (" + std::to_string(MAX_ALLOWED_FLOATS) +
		        "). Aborting.",
		    kcurrent_lib_name);

		return SocketReturnValue::kcount_too_large;
	}
	if (count == 0) {
		// end of file, empty block
		// arcforge::embedded::utils::Logger::GetInstance().Info("receiveFloat_safe: Received EOF marker (empty chunk)");
		arcforge::embedded::utils::Logger::GetInstance().Warning(
		    "receiveFloat_safe: Received EOF marker (empty chunk)", kcurrent_lib_name);
		return SocketReturnValue::keof;
	}

	// 5. receive data
	if (count > 0) {
		data.resize(count);
		size_t bytes_to_receive = count * sizeof(float);
		size_t bytes_has_received = 0;
		char* vector_start = reinterpret_cast<char*>(data.data());

		while (bytes_has_received < bytes_to_receive) {
			ssize_t n_recv = ::recv(socketfd_, vector_start + bytes_has_received,
			                        bytes_to_receive - bytes_has_received, 0);
			if (n_recv == 0) {
				// means received nothing
				arcforge::embedded::utils::Logger::GetInstance().Info(
				    "receiveFloat_safe: Peer closed connection while trying to receive count.",
				    kcurrent_lib_name);

				return SocketReturnValue::kpeer_abnormally_closed;
			}
			if (n_recv < 0) {
				// means something wrong when sending data

				arcforge::embedded::utils::Logger::GetInstance().Info(
				    "receiveFloat_safe: recv() error while receiving count. errno: " +
				        std::to_string(errno) + " (" + strerror(errno) + ")",
				    kcurrent_lib_name);
				return SocketReturnValue::kreceived_illegal;
			}

			bytes_has_received += static_cast<size_t>(n_recv);
		}
	}
	arcforge::embedded::utils::Logger::GetInstance().Info(
	    "receiveFloat_safe: Received " + std::to_string(count) + " floats.", kcurrent_lib_name);
	return SocketReturnValue::ksuccess;
}

// // --- sendString_safe  ---
// SocketReturnValue BaseImpl::sendString_safe(const std::string& message) {
// 	std::lock_guard<std::mutex> lock(*(socket_mutex_.get()));

// 	// 1. fd validation verification
// 	if (socketfd_ < 0) {
// 		return SocketReturnValue::kfd_illegal;
// 	}

// 	// 2. message length validation check
// 	uint32_t len = static_cast<uint32_t>(message.length());
// 	ssize_t first_n_send = ::send(socketfd_, &len, sizeof(len), 0);
// 	if (first_n_send != sizeof(len)) {
// 		arcforge::embedded::utils::Logger::GetInstance().Info("sendString_safe: Failed to send length.");
// 		return SocketReturnValue::ksendlength_failed;
// 	}

// 	// 3. send message its body
// 	if (len > 0) {
// 		if (::send(socketfd_, message.c_str(), len, 0) != static_cast<ssize_t>(len)) {
// 			arcforge::embedded::utils::Logger::GetInstance().Info("sendString_safe: Failed to send data.");
// 			return SocketReturnValue::ksenddata_failed;
// 		}
// 	}
// 	arcforge::embedded::utils::Logger::GetInstance().Info("sendString_safe: Sent string of length " + std::to_string(len) + ".");
// 	return SocketReturnValue::ksuccess;
// }
// --- sendString_safe  ---
SocketReturnValue BaseImpl::sendString_safe(const std::string& message) {
	std::lock_guard<std::mutex> lock(*(socket_mutex_.get()));

	if (socketfd_ < 0) {
		return SocketReturnValue::kfd_illegal;
	}

	uint32_t len = static_cast<uint32_t>(message.length());

	if (::send(socketfd_, &len, sizeof(len), 0) != sizeof(len)) {
		arcforge::embedded::utils::Logger::GetInstance().Info(
		    "sendString_safe: Failed to send length. errno: " + std::to_string(errno),
		    kcurrent_lib_name);
		return SocketReturnValue::ksendlength_failed;
	}

	if (len > 0) {
		size_t bytes_to_send = len;
		size_t bytes_has_sent = 0;
		const char* data_ptr = message.c_str();

		while (bytes_has_sent < bytes_to_send) {
			ssize_t n_sent =
			    ::send(socketfd_, data_ptr + bytes_has_sent, bytes_to_send - bytes_has_sent, 0);
			if (n_sent < 0) {
				arcforge::embedded::utils::Logger::GetInstance().Info(
				    "sendString_safe: send() error while sending data. errno: " +
				        std::to_string(errno),
				    kcurrent_lib_name);
				return SocketReturnValue::ksenddata_failed;
			}
			bytes_has_sent += static_cast<size_t>(n_sent);
		}
	}
	arcforge::embedded::utils::Logger::GetInstance().Info(
	    "sendString_safe: Sent string of length " + std::to_string(len) + ".", kcurrent_lib_name);
	return SocketReturnValue::ksuccess;
}

// --- receiveString_safe  ---
SocketReturnValue BaseImpl::receiveString_safe(std::string& message) {
	std::lock_guard<std::mutex> lock(*(socket_mutex_.get()));

	// 1. fd validation verification
	if (socketfd_ < 0) {
		return SocketReturnValue::kfd_illegal;
	}
	// 2. clean the final result container
	message.clear();

	// 3. receive length of message
	uint32_t len;
	ssize_t len_recv_bytes = ::recv(socketfd_, &len, sizeof(len), 0);

	if (len_recv_bytes == 0) {
		arcforge::embedded::utils::Logger::GetInstance().Info(
		    "receiveString_safe: Peer closed connection while trying to receive length.",
		    kcurrent_lib_name);
		return SocketReturnValue::kpeer_abnormally_closed;
	}
	if (len_recv_bytes < 0) {
		arcforge::embedded::utils::Logger::GetInstance().Info(
		    "receiveString_safe: recv() error while receiving length. errno: " +
		        std::to_string(errno) + " (" + strerror(errno) + ")",
		    kcurrent_lib_name);
		return SocketReturnValue::kreceived_illegal;
	}
	if (len_recv_bytes != sizeof(len)) {
		arcforge::embedded::utils::Logger::GetInstance().Info(
		    "receiveString_safe: Failed to receive full length header. Expected " +
		        std::to_string(sizeof(len)) + ", got " + std::to_string(len_recv_bytes),
		    kcurrent_lib_name);
		return SocketReturnValue::kreceivelength_failed;
	}
	//---------------------------------
	// uint32_t len;
	// if (::recv(socketfd_, &len, sizeof(len), 0) != sizeof(len)) {
	// 	arcforge::embedded::utils::Logger::GetInstance().Info("receiveString_safe: Failed to receive length or peer closed.");

	// 	// return SocketReturnValue::ksendlength_failed;
	// 	return SocketReturnValue::kpeer_abnormally_closed;
	// }

	// 4. if this is empty string, just return corresponding value
	if (len == 0) {
		arcforge::embedded::utils::Logger::GetInstance().Info(
		    "receiveString_safe: This is empty string, return now.", kcurrent_lib_name);
		return SocketReturnValue::kempty_string;
	}

	// 5. if has content, receive message body ifself
	if (len > 0) {
		const uint32_t MAX_ALLOWED_CHARS = 1024 * 1024;
		if (len > MAX_ALLOWED_CHARS) {
			arcforge::embedded::utils::Logger::GetInstance().Info(
			    "receiveString_safe: Received length (" + std::to_string(len) +
			        ") exceeds max allowed size. Aborting.",
			    kcurrent_lib_name);
			return SocketReturnValue::kcount_too_large;
		}

		message.resize(len);
		size_t bytes_has_received = 0;
		char* buffer_start = &message[0];

		while (bytes_has_received < len) {
			ssize_t n_recv =
			    ::recv(socketfd_, buffer_start + bytes_has_received, len - bytes_has_received, 0);
			if (n_recv == 0) {
				arcforge::embedded::utils::Logger::GetInstance().Info(
				    "receiveString_safe: Peer closed connection mid-stream. Expected " +
				        std::to_string(len) + " bytes, but got only " +
				        std::to_string(bytes_has_received),
				    kcurrent_lib_name);
				message.clear();
				return SocketReturnValue::kpeer_abnormally_closed;
			}
			if (n_recv < 0) {
				arcforge::embedded::utils::Logger::GetInstance().Info(
				    "receiveString_safe: recv() error mid-stream. errno: " + std::to_string(errno) +
				        " (" + strerror(errno) + ")",
				    kcurrent_lib_name);
				message.clear();
				return SocketReturnValue::kreceived_illegal;
			}
			bytes_has_received += static_cast<size_t>(n_recv);
		}
	}
	//-----------------------------------------
	// if (len > 0) {
	// 	message.resize(len);
	// 	ssize_t n_recv = ::recv(socketfd_, &message[0], len, 0);
	// 	if (n_recv != static_cast<ssize_t>(len)) {
	// 		arcforge::embedded::utils::Logger::GetInstance().Info("receiveString_safe: Failed to receive data or peer closed. Expected " +
	// 		    std::to_string(len) + ", got " + std::to_string(n_recv));
	// 		message.clear();

	// 		// return SocketReturnValue::ksenddata_failed;
	// 		return SocketReturnValue::kpeer_abnormally_closed;
	// 	}
	// }

	arcforge::embedded::utils::Logger::GetInstance().Info(
	    "receiveString_safe: Received string of length " + std::to_string(len) + ".",
	    kcurrent_lib_name);
	return SocketReturnValue::ksuccess;
}
/*===================================================
 * client-only public interface
 *===================================================*/
SocketReturnValue BaseImpl::connectToServer() {

	// create Unix domain socket
	int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sock_fd < 0) {
		return SocketReturnValue::kfd_illegal;
	}

	// socket path verification
	if (getSocketPath_safe().empty() == true) {
		// std::cerr << "Socket path is empty, pls set it before you connect to server." << std::endl;
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "Socket path is empty, pls set it before you connect to server.", kcurrent_lib_name);
		return SocketReturnValue::ksocketpath_empty;
	}

	// config server address
	struct sockaddr_un server_addr;
	memset(&server_addr, 0, sizeof(server_addr));
	server_addr.sun_family = AF_UNIX;
	strncpy(server_addr.sun_path, getSocketPath_safe().c_str(), sizeof(server_addr.sun_path) - 1);

	// establish connection to server
	if (connect(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
		// perror("Client connect failed");
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    std::string("Client connect failed") + strerror(errno), kcurrent_lib_name);
		close(sock_fd);
		return SocketReturnValue::kconnect_server_failed;
	}

	this->setFD_safe(sock_fd);

	return SocketReturnValue::ksuccess;
}
/*===================================================
 * server-only public interfaces
 *===================================================*/
SocketReturnValue BaseImpl::startServer(const size_t& timeout) {

	int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sock_fd < 0) {
		return SocketReturnValue::kfd_illegal;
	}

	// socket path verification
	if (getSocketPath_safe().empty() == true) {
		// std::cerr << "Socket path is empty, pls set it before you connect to server." << std::endl;
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "Socket path is empty, pls set it before you connect to server.", kcurrent_lib_name);
		close(sock_fd);
		return SocketReturnValue::ksocketpath_empty;
	}

	struct sockaddr_un addr;
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, getSocketPath_safe().c_str(), sizeof(addr.sun_path) - 1);

	unlink(getSocketPath_safe().c_str());

	if (bind(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
		close(sock_fd);
		return SocketReturnValue::kbind_error;
	}

/*----------------------------------
	 * start listening
     * SOMAXCONN means more request in queued that the performance under network
	 * poor connection and etc scenes may be enhanced
	 *--------------------------------- */
#if 1
	int queue_size = 1;
#else
	int queue_size = SOMAXCONN;
#endif
	if (listen(sock_fd, queue_size) < 0) {
		close(sock_fd);
		return SocketReturnValue::klisten_error;
	}
	// arcforge::embedded::utils::Logger::GetInstance().Warning("size of the listen() queue is " + std::to_string(queue_size));
	arcforge::embedded::utils::Logger::GetInstance().Debug(
	    "size of the listen() queue is " + std::to_string(queue_size), kcurrent_lib_name);

	// timeout configuration
	if (timeout != static_cast<size_t>(-1)) {
		struct timeval tval;

		// i want more high-performance opt, divide a size_t is terrible large memory requirement
		tval.tv_sec = static_cast<time_t>(timeout / 1000);
		tval.tv_usec = static_cast<time_t>(timeout % 1000) * 1000;
		int retval = ::setsockopt(sock_fd, SOL_SOCKET, SO_RCVTIMEO, &tval, sizeof(tval));
		if (retval < 0) {
			// i want more standard version that i can get char * string which was edited by posix
			// perror("setsockopt failed");
			arcforge::embedded::utils::Logger::GetInstance().Error("setsockopt failed",
			                                                       kcurrent_lib_name);
			close(sock_fd);
			return SocketReturnValue::ksetsocketopt_error;
		}

		std::stringstream ss;
		ss << "timeout has been set to " << tval.tv_sec << "." << tval.tv_usec / 1000 << "s";
		arcforge::embedded::utils::Logger::GetInstance().Info(ss.str(), kcurrent_lib_name);
	}

	this->setFD_safe(sock_fd);

	return SocketReturnValue::ksuccess;
}

SocketAcceptImplReturn::SocketAcceptImplReturn()
    : return_value_impl(SocketReturnValue::kinit_state), client_impl(nullptr) {};

SocketAcceptImplReturn::SocketAcceptImplReturn(SocketReturnValue value,
                                               std::unique_ptr<BaseImpl> client_ptr)
    : return_value_impl(value), client_impl(std::move(client_ptr)) {};

SocketAcceptImplReturn BaseImpl::acceptClient() {

	struct sockaddr_un client_addr;
	socklen_t client_len = sizeof(client_addr);

	int listening_fd = this->getFD_safe();
	if (listening_fd < 0) {
		// perror("Server accept: Invalid listening FD for server");
		arcforge::embedded::utils::Logger::GetInstance().Info(
		    std::string("Server accept: Invalid listening FD for server") + strerror(errno),
		    kcurrent_lib_name);

		SocketAcceptImplReturn sreturn;
		sreturn.return_value_impl = SocketReturnValue::kfd_illegal;
		return sreturn;
	}

	// accept the connection from client
	int client_fd = ::accept(listening_fd, (struct sockaddr*)&client_addr, &client_len);
	if (client_fd < 0) {
		if (errno == EAGAIN || errno == EWOULDBLOCK) {
			// means timeout
			// arcforge::embedded::utils::Logger::GetInstance().Info("accept() timeout, this is normal");

			SocketAcceptImplReturn sreturn;
			sreturn.return_value_impl = SocketReturnValue::kaccept_timeout;
			return sreturn;
			// return SocketReturnValue::kaccept_timeout;
		}

		SocketAcceptImplReturn sreturn;
		sreturn.return_value_impl = SocketReturnValue::kfd_illegal;
		return sreturn;
		// return SocketReturnValue::kfd_illegal;
	}

	auto client_connection = std::make_unique<BaseImpl>();
	client_connection->setFD_safe(client_fd);

	return {SocketReturnValue::ksuccess, std::move(client_connection)};
	// return SocketReturnValue::ksuccess;
}

SocketReturnValue BaseImpl::unlinkSocketPath() {

	if (getSocketPath_safe().empty()) {
		arcforge::embedded::utils::Logger::GetInstance().Warning(
		    "no need to unlink socket path, due to it is empty", kcurrent_lib_name);
		return SocketReturnValue::ksocketpath_empty;  // Or some other appropriate error
	}

	int retval = ::unlink(getSocketPath_safe().c_str());

	if (retval != 0) {
		// zero means success, so in this branch, means errors
		int error_code_access = errno;
		std::ostringstream errmsg;
		errmsg << "Failed to unlink " << getSocketPath_safe() << ": " << strerror(error_code_access)
		       << " (errno: " << error_code_access << ")\n";
		arcforge::embedded::utils::Logger::GetInstance().Warning(errmsg.str(), kcurrent_lib_name);

		return SocketReturnValue::kunknownerror;
	}

	std::stringstream ss;
	ss << "[ServerPID:" << getpid() << "] Removed existing socket file: " << getSocketPath_safe();
	arcforge::embedded::utils::Logger::GetInstance().Warning(ss.str(), kcurrent_lib_name);

	return SocketReturnValue::ksuccess;
}

}  // namespace network_socket
}  // namespace embedded
}  // namespace arcforge