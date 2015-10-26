
#include "extension.h"

void print_bytes(const void *object, size_t size) {
	size_t i;
	cout << "LSB...MSB" << endl;
	printf("[ ");
	for(i = 0; i < size; i++) {
		printf("%02x ", ((const unsigned char *) object)[i] & 0xff);
	}
	printf("]\n");
}


Ext::Ext(int debug_level, std::list<Device*>& _deviceList)
: dbg ((debug_level>0), "cyan", g_sdh_debug_log), deviceList(_deviceList) {}

Ext::~Ext() {}

void Ext::FlushInput(long timeout_us_first, long timeout_us_subsequent, cRS232& comm_interface) {
	int bytes_read, bytes_read_total = 0;
	long timeout_us = timeout_us_first;
	do {
		UInt8 byte[4096];
		try {
			bytes_read = comm_interface.Read( &byte, 4096, timeout_us, true );
		}
		catch ( cRS232Exception* e )
		{
			delete e;
			bytes_read = 0;
			// ignore timeout exception
			break;
		}
		bytes_read_total += bytes_read;
		timeout_us = timeout_us_subsequent;
	} while (bytes_read > 0);
	dbg << "ignoring " << bytes_read_total << " old bytes of garbage from device\n";
}

void Ext::WriteCommandDSA( UInt8 command, UInt8* payload, UInt16 payload_len, cRS232& comm_interface) throw (ExtException*) {
	cCRC_DSACON32m checksum;
	int bytes_written = 0;
	int len;

	// gcc knows how to allocate variable size arrays on the stack
	char buffer[ payload_len + 8 ]; // 8 = 3 (preamble) + 1 (command) + 2 (len) + 2 (CRC)
	buffer[0] = (UInt8) 0xaa;
	buffer[1] = (UInt8) 0xaa;
	buffer[2] = (UInt8) 0xaa;
	buffer[3] = command;
	buffer[4] = ((UInt8*) &payload_len)[0];
	buffer[5] = ((UInt8*) &payload_len)[1];

	if ( payload_len > 0)
	{
		// command and payload length are included in checksum (if used)
		checksum.AddByte( command );
		checksum.AddByte( buffer[4] );
		checksum.AddByte( buffer[5] );
	}
	unsigned int i;
	for ( i=0; i < payload_len; i++)
	{
		checksum.AddByte( payload[ i ] );
		buffer[ 6+i ] = payload[i];
	}

	if ( payload_len > 0)
	{
		// there is a payload, so the checksum is sent along with the data
		len = payload_len + 8;
		buffer[len-2] = checksum.GetCRC_LB();
		buffer[len-1] = checksum.GetCRC_HB();
	}
	else
	{
		// no payload => no checksum
		len = 6;
	}

	bytes_written = comm_interface.write(buffer, len);

	if ( bytes_written != len )
		throw new ExtException( cMsg( "Could only write %d/%d bytes to DSACON32m", bytes_written, len ) );
}

void Ext::WriteCommandSDH(string command, cRS232& comm_interface) throw (ExtException*) {
	const char *EOL="\r\n";
	uint bytes_written = 0;

	bytes_written = comm_interface.write(command.c_str(), 0);
	if(bytes_written != command.size() ){
		throw new ExtException( cMsg( "Could only write %d/%d bytes to SDH-2", bytes_written, command.size()) );
	}
	bytes_written = comm_interface.write(EOL, 0);
	if(bytes_written != 2) {
		throw new ExtException( cMsg( "Could only write %d/%d bytes to SDH-2", bytes_written, 2) );
	}
}


int Ext::ReadBytes(UInt8 *data, cRS232& comm_interface, int max_size) throw (ExtException*) {
	int bytes_read = 0;
	int len = 0;
	long timeout_us = 10000;

	while(true) {

		bytes_read = comm_interface.Read(data + len, 1, timeout_us, false);

		if(bytes_read > 0) {
			len += bytes_read;

			if (len >= max_size) {
				break;
			}
		} else {
			break;
		}
	}

	return len;
}

UInt32 Ext::ExtractSerialDSA(UInt8 *data, int len) {

	//DSA response data format

	// Byte
	// 0-2  preamble
	// 3    packet id
	// 4-5  size
	// 6-7  erro code
	// 8-11 serial

	// Example
	// aa aa aa 01 12 00 00 00 57 00 9b 1d

	int nb_preamble_bytes = 0;
	bool preamble_found = false;
	int start;
	for(int i = 0; i < len; i++) {
		if(data[i] == 0xaa) {
			nb_preamble_bytes++;
			dbg << "found valid preamble byte no " << nb_preamble_bytes << "\n";
		} else {
			nb_preamble_bytes = 0;
			dbg << "ignoring invalid preamble byte " << int(data[i]) << "\n";
		}
		if(nb_preamble_bytes == 3) {
			preamble_found = true;
			start = i-2;
			break;
		}
	}
	if(!preamble_found) {
		return -1;
	}

	UInt32 serial = -1;

	// validity checks
	if( (data[start+3] == 0x01) && (data[start+6] == 0x00) && (data[start+7] == 0x00) ) {
		// assume little endianess
		serial = 0;
		serial =                 data[start+11];
		serial = (serial << 8) + data[start+10];
		serial = (serial << 8) + data[start+9];
		serial = (serial << 8) + data[start+8];
	}
	return serial;
}

UInt32 Ext::ExtractSerialSDH(UInt8 *data, int len) {
	//SDH response data format
	// Example
	// ASCII response: 53 4e 3d 34 38 0d 0a
	// equivalent to:  SN=48

	UInt32 serial = -1;

	// validity checks
	if( (data[0] == 'S') && (data[1] == 'N') && (data[2] == '=') ) {
		serial = 10*(data[3]-48) + data[4]-48; // conversion from ASCII digits to decimal number
	}

	return serial;
}


void Ext::IdentifyDevices() {

	// Detect avaiable tty devices
	list<string> comportList = listComports();
	cout << "Available serial ports found: " <<  comportList.size() << "\n";
	for(list<string>::const_iterator comport = comportList.begin(); comport != comportList.end(); ++comport) {
		cout << *comport << "\n";
	}

	// Open tty* devices one after another and try to communicate to determine the actual device
	for(list<string>::const_iterator comport = comportList.begin(); comport != comportList.end(); ++comport) {
		dbg << "Probing Comport " << *comport << "\n";
		//cout << "Searching device on " << *comport << endl;
		cRS232 comm_interface(1, 115200, 1.0, (*comport).c_str());

		comm_interface.Open();

		//TODO: Set framerate of remote DSACON32m to 0 first.
		// For now: simply reboot hand

		FlushInput(10000, 1000, comm_interface);

		// Iterate over list of devices and analyze device response. Remove successful devices from device list
		for(list<Device*>::iterator it = deviceList.begin(); it != deviceList.end(); ++it) {
			if((*it)->device_format_string.empty()) { // not yet determined
				UInt32 serial = -1;
				int max_data = 1024;
				UInt8 data[max_data];
				int len = 0;

				if((*it)->device_type == SDH2) {
					WriteCommandSDH("sn", comm_interface);
					len = ReadBytes(data, comm_interface, max_data);
					serial = ExtractSerialSDH(data, len);
				}

				if((*it)->device_type == DSACON32m) {
					WriteCommandDSA(0x01, comm_interface);
					len = ReadBytes(data, comm_interface, max_data);
					serial = ExtractSerialDSA(data, len);
				}

				// Determine corresponding comport and remove device from to-do list
				if(serial == (*it)->serial_no) {
					cout << (*it)->device_name << " detected on " << *comport << endl;
					(*it)->device_format_string = *comport;
					break;
				}
			}
		}
		comm_interface.Close();
	}
}


string Ext::getComportDriver(const string& tty) {
	struct stat st;
	string devicedir = tty;

	// Append '/device' to the tty-path
	devicedir += "/device";

	// Stat the devicedir and handle it if it is a symlink
	if (lstat(devicedir.c_str(), &st)==0 && S_ISLNK(st.st_mode)) {
		char buffer[1024];
		memset(buffer, 0, sizeof(buffer));

		// Append '/driver' and return basename of the target
		devicedir += "/driver";

		if (readlink(devicedir.c_str(), buffer, sizeof(buffer)) > 0)
			return basename(buffer);
	}
	return "";
}

void  Ext::addComport(list<string>& comList, list<string>& comList8250, const string& dir) {
	// Get the driver the device is using
	string driver = getComportDriver(dir);

	// Skip devices without a driver
	if (driver.size() > 0) {
		string devfile = string("/dev/") + basename(dir.c_str());

		// Put serial8250-devices in a separate list
		if (driver == "serial8250") {
			comList8250.push_back(devfile);
		} else
			comList.push_back(devfile);
	}
}

void  Ext::probeSerial8250Comports(list<string>& comList, list<string> comList8250) {
	struct serial_struct serinfo;
	list<string>::iterator it = comList8250.begin();

	// Iterate over all serial8250-devices
	while (it != comList8250.end()) {

		// Try to open the device
		int fd = open((*it).c_str(), O_RDWR | O_NONBLOCK | O_NOCTTY);

		if (fd >= 0) {
			// Get serial_info
			if (ioctl(fd, TIOCGSERIAL, &serinfo)==0) {
				// If device type is no PORT_UNKNOWN we accept the port
				if (serinfo.type != PORT_UNKNOWN)
					comList.push_back(*it);
			}
			close(fd);
		}
		it ++;
	}
}

list<string>  Ext::listComports() {
	int n;
	struct dirent **namelist;
	list<string> comList;
	list<string> comList8250;
	const char* sysdir = "/sys/class/tty/";

	// Scan through /sys/class/tty - it contains all tty-devices in the system
	n = scandir(sysdir, &namelist, NULL, NULL);
	if (n < 0)
		perror("scandir");
	else {
		while (n--) {
			if (strcmp(namelist[n]->d_name,"..") && strcmp(namelist[n]->d_name,".")) {

				// Construct full absolute file path
				string devicedir = sysdir;
				devicedir += namelist[n]->d_name;

				// Register the device
				addComport(comList, comList8250, devicedir);
			}
			free(namelist[n]);
		}
		free(namelist);
	}

	// Only non-serial8250 has been added to comList without any further testing
	// serial8250-devices must be probe to check for validity
	probeSerial8250Comports(comList, comList8250);

	// Return the list of detected comports
	return comList;
}
