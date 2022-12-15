#include <iostream>
#include <vector>

using namespace std;

int main()
{
	vector<int>::iterator itr;
	vector<int> v = { 8, 4, 3, 2, 6, 7, 3, 1, 9, 4, 3, 0, 1, 2, 5, 4 };

	do {
		bool InvalidValue = true;
		bool valueErased = false;
		int x;
		cout << "Please enter a value to remove from the vector (enter 100 to quit): ";
		cin >> x;
		int currentvalue = 10;
		if (x != 100) {
			for (int j = 0; j < v.size(); j++) {
				if (v[j] == x) {
					if (!valueErased) {
						v.erase(v.begin() + j);
						valueErased = true;
						InvalidValue = false;
					}
				}
			}
			if (InvalidValue == true) {
				cout << "Value not found" << endl;
			}
			for (int i = 0; i < v.size(); i++) {
				cout << v[i] << ' ';
			}
			cout << endl;
		}
		else { exit(0);}
	} while (!v.empty());
}

