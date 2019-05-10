#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include "pc_crc16.h"
#include "final_aux.h"

// TODO: Add device name here
#define DEVICE_NAME "/dev/ttyUSB0"

int ifd, ofd;
struct termios oldtio, tio;
char* dev_name = DEVICE_NAME;
int8_t start_byte = MSG_START;

/* Initialize serial connection */
void init_serial() {
	// Open the serial port (/dev/ttyS1) read-write
	ifd = open(dev_name, O_RDWR | O_NOCTTY);
	if (ifd < 0) {
		perror(dev_name);
		exit(EXIT_FAILURE);
	}
	ofd = ifd;

	// Create settings for serial port communication to ABS
	tio.c_cflag = B9600 | CS8 | CLOCAL | CREAD;
	tio.c_iflag = 0;
	tio.c_oflag = 0;
	tio.c_lflag = 0;
	// Retrieve old I/O settings
	tcgetattr(ifd, &oldtio);
	// Flush existing I/O content and apply new attributes
	tcflush(ifd ,TCIFLUSH);
	tcsetattr(ifd, TCSANOW, &tio);
	return;
}

/* Close and resume serial device settings */
void end_serial() {
	tcflush(ifd, TCIFLUSH);
	tcsetattr(ifd, TCSANOW, &oldtio);
	close(ifd);
	return;
}

void send_facial_pos(double x_pos, double y_pos)
{
	char msg_position[MSG_BYTES_MSG];	
	char ack_buffer[1];
	unsigned short crc = 0;
	int8_t low, high = 0;
	int ack, attempts = 0;
	int write_ret, read_ret = 0;

	float x_approx = (float)x_pos;
	float y_approx = (float)y_pos;
	memcpy(msg_position, &x_approx, sizeof(x_approx));
	memcpy((msg_position + 4), &y_approx, sizeof(y_approx));

	// Reset states
	attempts = 0;
	
	crc = pc_crc16(msg_position, MSG_BYTES_MSG + 1);
	low = crc;				// Cast lowest 8 bits into a byte
	high = crc >> 8;			// Grab the higher 8 bits
	printf("crc = 0x%04x \n", crc);
	ack = MSG_NACK;  // Reset flag to NACK status
	while (ack != MSG_ACK)
	{
		read_ret = 0;
		write_ret = 0;
		printf("Sending (attempt %d)...\n", ++attempts);
		// Send out start byte
		write_ret += write(ofd, &start_byte, 1);
		write_ret += write(ofd, &high, 1);
		write_ret += write(ofd, &low, 1);
		write_ret += write(ofd, msg_position, MSG_BYTES_MSG);	
		printf("Message sent (%d bytes), waiting for ack...\n", write_ret);
		while (!read_ret) {
			read_ret += read(ifd, ack_buffer, 1);
		}

		/* 
		printf("crc = 0x%04x \n", crc);
		for (i = 0; i < 8; i++) {
			printf("%03d ", (int)(msg_position[i]));
		}
		printf("\n");
		*/
		
		// Convert buffer's content to ACK/NACK message
		ack = (int)ack_buffer[0];  // Assume we only receive either 0 or 1
		printf("%s\n", (ack == 1) ? "ACK" : "NACK, resending");
		if (ack == 3) {
			printf("Cause: Pack corrupted.\n");
		} else if (ack == 0) {
			printf("Cause: Timeout.\n");
		}
		
		// Stop message sending routine if ACK received
		if (ack == MSG_ACK) {
			break;
		}
	}	
	return;
}


