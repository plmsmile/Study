#include <iostream>


/**
 * 在数组中，查找a的重复数字，找出任意一个就行
 * Args:
 *  a: 无序数组，值的区间[0, n-1]
 *  length: 长度
 *  dup: 要返回的数值
 * Returns:
 *  数据合法并且有重复的数字，返回True；否则返回Fasle
 */
bool duplicate(int a[], int length, int *dup) {
    // 数据合法性检查
    if (a == nullptr || length < 0) {
        return false;
    }
    for (int i = 0; i < length; ++i) {
        if (a[i] < 0 || a[i] >= length) {
            return false;
        }
    }

    // 遍历数组，把i放到a[i]位置上
    for (int i = 0; i < length; ++i) {
        // 找到i放到a[i]上
        while (a[i] != i) {
            int m = a[i];
            if (a[m] == m) {
                *dup = m;
                return true;
            } else {
                // 把m放到a[m]上，交换
                a[i] = a[m];
                a[m] = m;
            }
        }
    }
    return false;
}

/**
 * 判断数组中是否包含某个数字
 * Args:
 *  a: 数组
 *  length: 数组长度
 *  target_num: 被检测的数字
 * Returns:
 *  存在true，不存在false
 */
bool contains(int a[], int length, int target_num) {
    for (int i = 0; i < length; i++) {
        if (a[i] == target_num) {
            return true;
        }
    }
    return false;
}

void test(const char* test_name, int a[], int length, int dups[], int dups_len, bool valid_arg) {
    int dup;
    bool res = duplicate(a, length, &dup);
    std::cout << test_name << std::endl;
    if (res == valid_arg) {
        if (res == true) {
            if (true == contains(dups, dups_len, dup)) {
                std::cout << "PASS" << std::endl;
            } else {
                std::cout << "FAILED" << std::endl;
            }
        } else {
            std::cout << "PASS" << std::endl;
        }
    } else {
        std::cout << "Failed" << std::endl;
    }
}

void test1() {
    int a[] = {2, 4, 3, 1, 4};
    int dups[] = {2, 4};
    test("test1", a, sizeof(a)/sizeof(int), dups, sizeof(a)/sizeof(int), true);
}


int main() {
    test1();
    return 0;
}
