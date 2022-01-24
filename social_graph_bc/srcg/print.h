#pragma once
#include <iostream>
#include <stdio.h>

__device__
size_t __strlen(const char *str)
{
	const char *s;
	for (s = str; *s; ++s);
	return (s - str);
}

__device__
void __itoa(int num, char* str)
{
	const int base = 10;
	int i = 0;
	bool isNegative = false;

	if (num == 0)
	{
		str[i++] = '0';
		str[i] = '\0';
		return;
	}

	if (num < 0 && base == 10)
	{
		isNegative = true;
		num = -num;
	}

	while (num != 0)
	{
		int rem = num % base;
		str[i++] = (rem > 9) ? (rem - 10) + 'a' : rem + '0';
		num = num / base;
	}

	if (isNegative)
		str[i++] = '-';

	str[i] = '\0'; // Append string terminator

	// Reverse the string
	char tmp;
	for (int index = 0; index < i/2; ++index)
	{
		if ((i - index - 1) < 0)
			break;

		tmp = str[index];
		str[index] = str[i - index - 1];
		str[i - index - 1] = tmp;
	}
}

template <typename T>
__device__
void print(T* data, size_t size)
{
	char buffer[2000] = { 0 };
	char value[6];

	size_t count = 0, value_len = 0;
	for (size_t i = 0; i < size; ++i)
	{
		__itoa(data[i], value);
		value_len = __strlen(value);

		memcpy(&buffer[count], value, value_len);
		count += value_len;
		buffer[count++] = ',';
	}

	//fprintf(stdout,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

	printf("{%s}\n", buffer);
}

/*
int main()
{
	short arr[] = { 100, 200, 331, 513, 642, 431, 676 };

	print(arr, 7);
	
	system("pause");
	return 0;
}
*/
