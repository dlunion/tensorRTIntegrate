
#include "cc_util.hpp"
#include <fstream>
#include <stack>
#include <algorithm>
#include <mutex>
#include <memory>
#include <thread>

#if defined(U_OS_WINDOWS)
#	define HAS_UUID
#	include <Windows.h>
#	include <Shlwapi.h>
#	pragma comment(lib, "shlwapi.lib")
#   pragma comment(lib, "ole32.lib")
#   pragma comment(lib, "gdi32.lib")
#	undef min
#	undef max
#endif

#if defined(U_OS_LINUX)
#	include <sys/io.h>
#	include <dirent.h>
#	include <sys/types.h>
#	include <sys/stat.h>
#	include <unistd.h>
#   include <stdarg.h>
#if defined(HAS_UUID)
//sudo apt-get install uuid-dev
#   include <uuid/uuid.h>
#endif

#	define strtok_s  strtok_r
#endif

namespace ccutil{

	using namespace std;

#if defined(U_OS_WINDOWS)

#define _LLFMT	"%I64"
#define _TOSTRING(buf, fmt, val)	\
	sprintf_s(buf, sizeof (buf), fmt, val)
#else

#define _LLFMT	"%ll"
#define _TOSTRING(buf, fmt, val)	\
	snprintf(buf, sizeof (buf), fmt, val)
#endif

	string tostr(int val){
		char buffer[2 * 32];

		_TOSTRING(buffer, "%d", val);
		return buffer;
	}

	string tostr(unsigned int val){
		char buffer[2 * 32];

		_TOSTRING(buffer, "%u", val);
		return buffer;
	}

	string tostr(long val){
		char buffer[2 * 32];

		_TOSTRING(buffer, "%ld", val);
		return buffer;
	}

	string tostr(unsigned long val){
		char buffer[2 * 32];

		_TOSTRING(buffer, "%lu", val);
		return buffer;
	}

	string tostr(long long val){
		char buffer[2 * 32];

		_TOSTRING(buffer, _LLFMT "d", val);
		return buffer;
	}

	string tostr(unsigned long long val){
		char buffer[2 * 32];

		_TOSTRING(buffer, _LLFMT "u", val);
		return buffer;
	}

	string tostr(const void* val){
		char buffer[64];
		_TOSTRING(buffer, "%p", val);
		return buffer;
	}

	string tostr(long double val)
	{	// convert long double to string
		typedef back_insert_iterator<string> _Iter;
		typedef num_put<char, _Iter> _Nput;
		const _Nput& _Nput_fac = use_facet<_Nput>(locale());
		ostream _Ios((streambuf *)0);
		string str;

		_Ios.setf(ios_base::fixed);
		_Nput_fac.put(_Iter(str), _Ios, ' ', val);
		return str;
	}

	string tostr(double val){
		return (tostr((long double)val));
	}

	string tostr(float val){
		return (tostr((long double)val));
	}

	static const char* level_string(int level){
		switch (level){
		case LINFO: return "I";
		case LWARNING: return "W";
		case LERROR: return "E";
		case LFATAL: return "F";
		default: return "Unknow";
		}
	}

	AssertStream::AssertStream(bool condition, const char* file, int line, const char* code) :
		condition_(condition), file_(file), line_(line), code_(code){
	}

	AssertStream::AssertStream(){
		condition_ = true;
	}

	AssertStream::~AssertStream(){
		if (!condition_){
			if (msg_.empty()){
				__log_func(file_, line_, LFATAL, "Assert Failure: %s", code_);
			}
			else{
				__log_func(file_, line_, LFATAL, "Assert Failure: %s, %s", code_, msg_.c_str());
			}
			abort();
		}
	}

	/////////////////////////////////////////////////////////////////////////////
	LoggerStream::LoggerStream(bool condition, int level, const char* file, int line) :
		condition_(condition), level_(level), file_(file), line_(line){
	}

	LoggerStream::~LoggerStream(){

		if (condition_){
			if (msg_.empty())
				__log_func(file_, line_, level_, "");
			else
				__log_func(file_, line_, level_, "%s", msg_.c_str());
		}

		if (level_ == LFATAL){
			abort();
		}
	}

	/////////////////////////////////////////////////////////////////////////////
	static shared_ptr<cv::RNG>& getRandom(){

		static shared_ptr<cv::RNG> g_random;
		static volatile bool g_isinited = false;

		if (g_isinited)
			return g_random;

		g_isinited = true;
		g_random.reset(new cv::RNG(25));
		return g_random;
	}

	Timer::Timer(){
		begin();
	}

	void Timer::begin(){
		tick = cv::getTickCount();
	}

	double Timer::end(){	//ms

		double fee = (cv::getTickCount() - tick) / cv::getTickFrequency() * 1000;

		begin();
		return fee;
	}

	int GenNumber::next(){
		return next_++;
	}

	BBox::BBox(const cv::Rect& other){
		x = other.x;
		y = other.y;
		r = other.x + other.width - 1;
		b = other.y + other.height - 1;
		score = 0;
	}

	BBox BBox::offset(const cv::Point& position) const{

		BBox r(*this);
		r.x += position.x;
		r.y += position.y;
		r.r += position.x;
		r.b += position.y;
		return r;
	}

	cv::Point BBox::tl() const{
		return cv::Point(x, y);
	}

	cv::Point BBox::rb() const{
		return cv::Point(r, b);
	}

	float BBox::width() const{
		return (r - x) + 1;
	}

	float BBox::height() const{
		return (b - y) + 1;
	}

	cv::Point2f BBox::center() const {
		return cv::Point2f((x + r) * 0.5, (y + b) * 0.5);
	}

	float BBox::area() const{
		return width() * height();
	}

	BBox::BBox(){
	}

	BBox::BBox(float x, float y, float r, float b, float score, int label) :
		x(x), y(y), r(r), b(b), score(score), label(label){
	}

	BBox::operator cv::Rect() const{
		return box();
	}

	cv::Rect BBox::box() const{
		return cv::Rect(x, y, width(), height());
	}

	BBox BBox::transfrom(cv::Size sourceSize, cv::Size dstSize){

		auto& a = *this;
		BBox out;
		out.x = a.x / (float)sourceSize.width * dstSize.width;
		out.y = a.y / (float)sourceSize.height * dstSize.height;
		out.r = a.r / (float)sourceSize.width * dstSize.width;
		out.b = a.b / (float)sourceSize.height * dstSize.height;
		return out;
	}

	BBox BBox::mergeOf(const BBox& b) const{
		auto& a = *this;
		BBox out;
		out.x = min(a.x, b.x);
		out.y = min(a.y, b.y);
		out.r = max(a.r, b.r);
		out.b = max(a.b, b.b);
		return out;
	}

	BBox BBox::expandMargin(float margin, const cv::Size& limit) const {

		BBox expandbox;
		expandbox.x = (int)(this->x - margin);
		expandbox.y = (int)(this->y - margin);
		expandbox.r = (int)(this->r + margin);
		expandbox.b = (int)(this->b + margin);

		if (limit.area() > 0)
			expandbox = expandbox.box() & cv::Rect(0, 0, limit.width, limit.height);
		return expandbox;
	}

	BBox BBox::expand(float ratio, const cv::Size& limit) const{

		BBox expandbox;
		expandbox.x = (int)(this->x - this->width() * ratio);
		expandbox.y = (int)(this->y - this->height() * ratio);
		expandbox.r = (int)(this->r + this->width() * ratio);
		expandbox.b = (int)(this->b + this->height() * ratio);
		
		if (limit.area() > 0)
			expandbox = expandbox.box() & cv::Rect(0, 0, limit.width, limit.height);
		return expandbox;
	}

	float BBox::iouMinOf(const BBox& b) const{
		auto& a = *this;
		float xmax = max(a.x, b.x);
		float ymax = max(a.y, b.y);
		float xmin = min(a.r, b.r);
		float ymin = min(a.b, b.b);
		float uw = (xmin - xmax + 1 > 0) ? (xmin - xmax + 1) : 0;
		float uh = (ymin - ymax + 1 > 0) ? (ymin - ymax + 1) : 0;
		float iou = uw * uh;
		return iou / min(a.area(), b.area());
	}

	float BBox::iouOf(const BBox& b) const{

		auto& a = *this;
		float xmax = max(a.x, b.x);
		float ymax = max(a.y, b.y);
		float xmin = min(a.r, b.r);
		float ymin = min(a.b, b.b);
		float uw = (xmin - xmax + 1 > 0) ? (xmin - xmax + 1) : 0;
		float uh = (ymin - ymax + 1 > 0) ? (ymin - ymax + 1) : 0;
		float iou = uw * uh;
		return iou / (a.area() + b.area() - iou);
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	bool alphabetEqual(char a, char b, bool ignore_case){
		if (ignore_case){
			a = a > 'a' && a < 'z' ? a - 'a' + 'A' : a;
			b = b > 'a' && b < 'z' ? b - 'a' + 'A' : b;
		}
		return a == b;
	}

	static bool patternMatchBody(const char* str, const char* matcher, bool igrnoe_case){
		//   abcdefg.pnga          *.png      > false
		//   abcdefg.png           *.png      > true
		//   abcdefg.png          a?cdefg.png > true

		if (!matcher || !*matcher || !str || !*str) return false;

		const char* ptr_matcher = matcher;
		while (*str){
			if (*ptr_matcher == '?'){
				ptr_matcher++;
			}
			else if (*ptr_matcher == '*'){
				if (*(ptr_matcher + 1)){
					if (patternMatchBody(str, ptr_matcher + 1, igrnoe_case))
						return true;
				}
				else{
					return true;
				}
			}
			else if (!alphabetEqual(*ptr_matcher, *str, igrnoe_case)){
				return false;
			}
			else{
				if (*ptr_matcher)
					ptr_matcher++;
				else
					return false;
			}
			str++;
		}

		while (*ptr_matcher){
			if (*ptr_matcher != '*')
				return false;
			ptr_matcher++;
		}
		return true;
	}

	bool patternMatch(const char* str, const char* matcher, bool igrnoe_case){
		//   abcdefg.pnga          *.png      > false
		//   abcdefg.png           *.png      > true
		//   abcdefg.png          a?cdefg.png > true

		if (!matcher || !*matcher || !str || !*str) return false;

		char filter[500];
		strcpy(filter, matcher);

		vector<const char*> arr;
		char* ptr_str = filter;
		char* ptr_prev_str = ptr_str;
		while (*ptr_str){
			if (*ptr_str == ';'){
				*ptr_str = 0;
				arr.push_back(ptr_prev_str);
				ptr_prev_str = ptr_str + 1;
			}
			ptr_str++;
		}

		if (*ptr_prev_str)
			arr.push_back(ptr_prev_str);

		for (int i = 0; i < arr.size(); ++i){
			if (patternMatchBody(str, arr[i], igrnoe_case))
				return true;
		}
		return false;
	}

	vector<string> split(const string& str, const std::string& spstr){

		vector<string> res;
		if (str.empty()) return res;
		if (spstr.empty()) return{ str };

		auto p = str.find(spstr);
		if (p == string::npos) return{ str };

		res.reserve(5);
		string::size_type prev = 0;
		int lent = spstr.length();
		const char* ptr = str.c_str();

		while (p != string::npos){
			int len = p - prev;
			if (len > 0){
				res.emplace_back(str.substr(prev, len));
			}
			prev = p + lent;
			p = str.find(spstr, prev);
		}

		int len = str.length() - prev;
		if (len > 0){
			res.emplace_back(str.substr(prev, len));
		}
		return res;
	}

	vector<int> splitInt(const string& str, const string& spstr){

		auto arr = split(str, spstr);
		vector<int> out(arr.size());
		for (int i = 0; i < arr.size(); ++i)
			out[i] = atoi(arr[i].c_str());
		return out;
	}

	vector<float> splitFloat(const string& str, const string& spstr){
		auto arr = split(str, spstr);
		vector<float> out(arr.size());
		for (int i = 0; i < arr.size(); ++i)
			out[i] = atof(arr[i].c_str());
		return out;
	}

	vector<string> findFilesAndCacheList(const string& directory, const string& filter, bool findDirectory, bool includeSubDirectory){

		string path = directory;
		if (path.empty()) return vector<string>();
#ifdef U_OS_WINDOWS
		if (path.back() == '/' || path.back() == '\\'){
#endif

#ifdef U_OS_LINUX
			if (path.back() == '/'){
#endif
			path.pop_back();
		};

		string dirname = fileName(path);
		string findEncode = md5(directory + ";" + filter + ";" + (findDirectory ? "yes" : "no") + ";" + (includeSubDirectory ? "yes" : "no"));
		string cacheFile = dirname + "_" + findEncode + ".list.txt";

		vector<string> files;
		if (!ccutil::exists(cacheFile)){
			files = ccutil::findFiles(directory, filter, findDirectory, includeSubDirectory);
			ccutil::shuffle(files);
			ccutil::saveList(cacheFile, files);
		}
		else{
			files = ccutil::loadList(cacheFile);
		}
		return files;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef U_OS_WINDOWS
	vector<string> findFiles(const string& directory, const string& filter, bool findDirectory, bool includeSubDirectory){
		
		string realpath = directory;
		if (realpath.empty())
			realpath = "./";

		char backchar = realpath.back();
		if (backchar != '\\' && backchar != '/')
			realpath += "/";

		vector<string> out;
		_WIN32_FIND_DATAA find_data;
		stack<string> ps;
		ps.push(realpath);

		while (!ps.empty())
		{
			string search_path = ps.top();
			ps.pop();

			HANDLE hFind = FindFirstFileA((search_path + "*").c_str(), &find_data);
			if (hFind != INVALID_HANDLE_VALUE){
				do{
					if (strcmp(find_data.cFileName, ".") == 0 || strcmp(find_data.cFileName, "..") == 0)
						continue;

					if (!findDirectory && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != FILE_ATTRIBUTE_DIRECTORY ||
						findDirectory && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY){
						if (PathMatchSpecA(find_data.cFileName, filter.c_str()))
							out.push_back(search_path + find_data.cFileName);
					}

					if (includeSubDirectory && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY)
						ps.push(search_path + find_data.cFileName + "/");

				} while (FindNextFileA(hFind, &find_data));
				FindClose(hFind);
			}
		}
		return out;
	}
#endif

#ifdef U_OS_LINUX
	vector<string> findFiles(const string& directory, const string& filter, bool findDirectory, bool includeSubDirectory)
	{
		string realpath = directory;
		if (realpath.empty())
			realpath = "./";

		char backchar = realpath.back();
		if (backchar != '\\' && backchar != '/')
			realpath += "/";

		struct dirent* fileinfo;
		DIR* handle;
		stack<string> ps;
		vector<string> out;
		ps.push(realpath);

		while (!ps.empty())
		{
			string search_path = ps.top();
			ps.pop();

			handle = opendir(search_path.c_str());
			if (handle != 0)
			{
				while (fileinfo = readdir(handle))
				{
					struct stat file_stat;
					if (strcmp(fileinfo->d_name, ".") == 0 || strcmp(fileinfo->d_name, "..") == 0)
						continue;

					if (lstat((search_path + fileinfo->d_name).c_str(), &file_stat) < 0)
						continue;

					if (!findDirectory && !S_ISDIR(file_stat.st_mode) ||
						findDirectory && S_ISDIR(file_stat.st_mode))
					{
						if (patternMatch(fileinfo->d_name, filter.c_str()))
							out.push_back(search_path + fileinfo->d_name);
					}

					if (includeSubDirectory && S_ISDIR(file_stat.st_mode))
						ps.push(search_path + fileinfo->d_name + "/");
				}
				closedir(handle);
			}
		}
		return out;
	}
#endif
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	string loadfile(const string& file){

		ifstream in(file, ios::in | ios::binary);
		if (!in.is_open())
			return "";

		in.seekg(0, ios::end);
		size_t length = in.tellg();

		string data;
		if (length > 0){
			in.seekg(0, ios::beg);
			data.resize(length);

			in.read(&data[0], length);
		}
		in.close();
		return data;
	}

	size_t fileSize(const string& file){

#if defined(U_OS_LINUX)
		struct stat st;
		stat(file.c_str(), &st);
		return st.st_size;
#elif defined(U_OS_WINDOWS)
		WIN32_FIND_DATAA find_data;
		HANDLE hFind = FindFirstFileA(file.c_str(), &find_data);
		if (hFind == INVALID_HANDLE_VALUE)
			return 0;

		FindClose(hFind);
		return (uint64_t)find_data.nFileSizeLow | ((uint64_t)find_data.nFileSizeHigh << 32);
#endif
	}

	bool savefile(const string& file, const string& data, bool mk_dirs){
		return savefile(file, data.data(), data.size(), mk_dirs);
	}

	string format(const char* fmt, ...) {
		va_list vl;
		va_start(vl, fmt);
		char buffer[10000];
		vsprintf(buffer, fmt, vl);
		return buffer;
	}

	bool savefile(const string& file, const void* data, size_t length, bool mk_dirs){

		if (mk_dirs){
			int p = (int)file.rfind('/');

#ifdef U_OS_WINDOWS
			int e = (int)file.rfind('\\');
			p = max(p, e);
#endif
			if (p != -1){
				if (!mkdirs(file.substr(0, p)))
					return false;
			}
		}

		FILE* f = fopen(file.c_str(), "wb");
		if (!f) return false;

		if (data && length > 0){
			if (fwrite(data, 1, length, f) != length){
				fclose(f);
				return false;
			}
		}
		fclose(f);
		return true;
	}

	string middle(const string& str, const string& begin, const string& end){

		auto p = str.find(begin);
		if (p == string::npos) return "";
		p += begin.length();

		auto e = str.find(end, p);
		if (e == string::npos) return "";

		return str.substr(p, e - p);
	}

	bool savexml(const string& file, int width, int height, const vector<LabBBox>& objs){
		FILE* f = fopen(file.c_str(), "wb");
		if (!f) return false;

		fprintf(f, "<annotation>\n<size><width>%d</width><height>%d</height></size>\n", width, height);
		for (int i = 0; i < objs.size(); ++i){
			auto& obj = objs[i];
			fprintf(f,
				"<object>"
				"<name>%s</name>"
				"<bndbox>"
				"<xmin>%d</xmin>"
				"<ymin>%d</ymin>"
				"<xmax>%d</xmax>"
				"<ymax>%d</ymax>"
				"</bndbox>"
				"</object>\n", obj.classname.c_str(), (int)obj.x, (int)obj.y, (int)obj.r, (int)obj.b);
		}
		fprintf(f, "</annotation>");
		fclose(f);
		return true;
	}

	vector<LabBBox> loadxmlFromData(const string& data, int* width, int* height, const string& filter){

		vector<LabBBox> output;
		if (data.empty())
			return output;

		if (width)
			*width = atoi(middle(data, "<width>", "</width>").c_str());

		if (height)
			*height = atoi(middle(data, "<height>", "</height>").c_str());

		string begin_token = "<object>";
		string end_token = "</object>";
		int p = data.find(begin_token);
		if (p == -1)
			return output;

		bool ignoreFilter = filter.empty() || filter == "*";
		int e = data.find(end_token, p + begin_token.length());
		while (e != -1){

			string part = data.substr(p, e - p);
			string name = middle(part, "<name>", "</name>");

			//filter.empty, not use filter
			//filter == *, not use filter
			//filter == *xxx*, use match
			if (ignoreFilter || patternMatch(name.c_str(), filter.c_str())){
				float xmin = atof(middle(part, "<xmin>", "</xmin>").c_str());
				float ymin = atof(middle(part, "<ymin>", "</ymin>").c_str());
				float xmax = atof(middle(part, "<xmax>", "</xmax>").c_str());
				float ymax = atof(middle(part, "<ymax>", "</ymax>").c_str());

				LabBBox box;
				box.x = xmin;
				box.y = ymin;
				box.r = xmax;
				box.b = ymax;
				box.classname = name;
				output.push_back(box);
				//output.emplace_back(xmin, ymin, xmax, ymax, "", name);
			}

			e += end_token.length();
			p = data.find(begin_token, e);
			if (p == -1) break;

			e = data.find(end_token, p + begin_token.length());
		}
		return output;
	}

	vector<LabBBox> loadxml(const string& file, int* width, int* height, const string& filter){
		return loadxmlFromData(loadfile(file), width, height, filter);
	}

	bool xmlEmpty(const string& file){
		return loadxml(file).empty();
	}

	bool xmlHasObject(const string& file, const string& classes){
		auto objs = loadxml(file);
		for (int i = 0; i < objs.size(); ++i){
			if (objs[i].classname == classes)
				return true;
		}
		return false;
	}

	bool exists(const string& path){

#ifdef U_OS_WINDOWS
		return ::PathFileExistsA(path.c_str());
#elif defined(U_OS_LINUX)
		return access(path.c_str(), R_OK) == 0;
#endif
	}

	map<string, string> loadListMap(const string& listfile){

		auto list = loadList(listfile);
		map<string, string> mapper;
		if (list.empty())
			return mapper;

		string key;
		string value;
		for (int i = 0; i < list.size(); ++i){
			auto& line = list[i];
			int p = line.find(',');
			
			if (p == -1){
				key = line;
				value = "";
			}
			else{
				key = line.substr(0, p);
				value = line.substr(p + 1);
			}

			if (mapper.find(key) != mapper.end()){
				printf("repeat key: %s, existsValue: %s, newValue: %s\n", key.c_str(), mapper[key].c_str(), value.c_str());
			}
			mapper[key] = value;
		}
		return mapper;
	}

	vector<string> loadList(const string& listfile){

		vector<string> lines;
		string data = loadfile(listfile);

		if (data.empty())
			return lines;

		char* ptr = (char*)&data[0];
		char* prev = ptr;
		string line;

		while (true){
			if (*ptr == '\n' || *ptr == 0){
				int length = ptr - prev;

				if (length > 0){
					if (length == 1){
						if (*prev == '\r')
							length = 0;
					}
					else {
						if (prev[length - 1] == '\r')
							length--;
					}
				}

				if (length > 0){
					line.assign(prev, length);
					lines.push_back(line);
				}

				if (*ptr == 0)
					break;

				prev = ptr + 1;
			}
			ptr++;
		}
		return lines;
	}

	string directory(const string& path){

		if (path.empty())
			return "";

		int p = path.rfind('/');

#ifdef U_OS_WINDOWS
		int e = path.rfind('\\');
		p = max(p, e);
#endif
		return path.substr(0, p + 1);
	}

	bool saveList(const string& file, const vector<string>& list){

		FILE* f = fopen(file.c_str(), "wb");
		if (!f) return false;

		for (int i = 0; i < list.size(); ++i){

			auto& item = list[i];
			if (i < (int)list.size() - 1){
				fprintf(f, "%s\n", item.c_str());
			}
			else{
				fprintf(f, "%s", item.c_str());
			}
		}

		fclose(f);
		return true;
	}

	bool beginsWith(const string& str, const string& with){

		if (str.length() < with.length())
			return false;
		return strncmp(str.c_str(), with.c_str(), with.length()) == 0;
	}

	bool endsWith(const string& str, const string& with){

		if (str.length() < with.length())
			return false;

		return strncmp(str.c_str() + str.length() - with.length(), with.c_str(), with.length()) == 0;
	}

	string repstrFast(const string& str, const string& token, const string& value){

		string opstr;

		if (value.length() > token.length()){
			float numToken = str.size() / (float)token.size();
			float newTokenLength = value.size() * numToken;
			opstr.resize(newTokenLength);
		}
		else{
			opstr.resize(str.size());
		}

		char* dest = &opstr[0];
		const char* src = str.c_str();
		string::size_type pos = 0;
		string::size_type prev = 0;
		size_t token_length = token.length();
		size_t value_length = value.length();
		const char* value_ptr = value.c_str();
		bool keep = true;

		do{
			pos = str.find(token, pos);
			if (pos == string::npos){
				keep = false;
				pos = str.length();
			}

			size_t copy_length = pos - prev;
			memcpy(dest, src + prev, copy_length);
			dest += copy_length;
			
			if (keep){
				pos += token_length;
				prev = pos;
				memcpy(dest, value_ptr, value_length);
				dest += value_length;
			}
		} while (keep);

		size_t valid_size = dest - &opstr[0];
		opstr.resize(valid_size);
		return opstr;
	}

	string repstr(const string& str_, const string& token, const string& value){

		string str = str_;
		string::size_type pos = 0;
		string::size_type tokenlen = token.size();
		string::size_type vallen = value.size();

		while ((pos = str.find(token, pos)) != string::npos){
			str.replace(pos, tokenlen, value);
			pos += vallen;
		}
		return str;
	}

	bool isblank(const string& str, char blank){
		if (str.empty()) return true;
		
		const char* s = str.c_str();
		int len = str.length();
		for (int i = 0; i < len; ++i, ++s){
			if (*s != blank)
				return false;
		}
		return true;
	}

	void rmblank(vector<string>& list){

		vector<string> result;
		result.reserve(list.size());

		for (int i = 0; i < list.size(); ++i){
			if (!isblank(list[i]))
				result.push_back(list[i]);
		}
		std::swap(result, list);
		
		//for (int i = (int)list.size() - 1; i >= 0; --i){
		//	if (isblank(list[i]))
		//		list.erase(list.begin() + i);
		//}
	}

	string rmsuffix(const string& path){

		int p = path.rfind('.');
		if (p == -1)
			return path;
		
		int l = path.rfind('/');

#ifdef U_OS_WINDOWS
		int e = path.rfind('\\');
		l = max(l, e);
#endif
		if (p > l)
			return path.substr(0, p);
		return path;
	}

	string vocxml(const string& vocjpg){
		return repsuffix(repstr(vocjpg, "JPEGImages", "Annotations"), "xml");
	}

	string vocjpg(const string& vocxml){
		return repsuffix(repstr(vocxml, "Annotations", "JPEGImages"), "jpg");
	}

	string repsuffix(const string& path, const string& newSuffix){

		int p = path.rfind('.');
		if (p == -1)
			return path + "." + newSuffix;

		int l = path.rfind('/');

#ifdef U_OS_WINDOWS
		int e = path.rfind('\\');
		l = max(l, e);
#endif
		if (p > l)
			return path.substr(0, p + 1) + newSuffix;

		//没有.的文件，只是在尾巴加后缀，这种有点是在路径上的点而不是文件名的点
		return path + "." + newSuffix;
	}

	vector<string> batchRepSuffix(const vector<string>& filelist, const string& newSuffix){
		
		vector<string> newlist = filelist;
		auto lambda = [&](string& file){file = repsuffix(file, newSuffix); };
		each(newlist, lambda);
		return newlist;
	}

	string fileName(const string& path, bool include_suffix){

		if (path.empty()) return "";

		int p = path.rfind('/');

#ifdef U_OS_WINDOWS
		int e = path.rfind('\\');
		p = max(p, e);
#endif
		p += 1;

		//include suffix
		if (include_suffix)
			return path.substr(p);

		int u = path.rfind('.');
		if (u == -1)
			return path.substr(p);

		if (u <= p) u = path.size();
		return path.substr(p, u - p);
	}

	vector<BBox> nmsAsClass(const vector<BBox>& objs, float iou_threshold) {

		map<int, vector<BBox>> mapper;
		for (int i = 0; i < objs.size(); ++i) {
			mapper[objs[i].label].push_back(objs[i]);
		}

		vector<BBox> out;
		for (auto& item : mapper) {
			auto& objsClasses = item.second;
			std::sort(objsClasses.begin(), objsClasses.end(), [](const BBox& a, const BBox& b) {
				return a.score > b.score;
			});
			
			vector<int> flags(objsClasses.size());
			for (int i = 0; i < objsClasses.size(); ++i) {
				if (flags[i] == 1) continue;

				out.push_back(objsClasses[i]);
				flags[i] = 1;
				for (int k = i + 1; k < objsClasses.size(); ++k) {
					if (flags[k] == 0) {
						float iouUnion = objsClasses[i].iouOf(objsClasses[k]);
						if (iouUnion > iou_threshold)
							flags[k] = 1;
					}
				}
			}
		}
		return out;
	}

	vector<BBox> nms(vector<BBox>& objs, float iou_threshold){

		std::sort(objs.begin(), objs.end(), [](const BBox& a, const BBox& b){
			return a.score > b.score;
		});

		vector<BBox> out;
		vector<int> flags(objs.size());
		for (int i = 0; i < objs.size(); ++i){
			if (flags[i] == 1) continue;

			out.push_back(objs[i]);
			flags[i] = 1;
			for (int k = i + 1; k < objs.size(); ++k){
				if (flags[k] == 0){
					float iouUnion = objs[i].iouOf(objs[k]);
					if (iouUnion > iou_threshold)
						flags[k] = 1;
				}
			}
		}
		return out;
	}

	vector<BBox> nmsMinIoU(vector<BBox>& objs, float iou_threshold){

		std::sort(objs.begin(), objs.end(), [](const BBox& a, const BBox& b){
			return a.score > b.score;
		});

		vector<BBox> out;
		vector<int> flags(objs.size());
		for (int i = 0; i < objs.size(); ++i){
			if (flags[i] == 1) continue;

			out.push_back(objs[i]);
			flags[i] = 1;
			for (int k = i + 1; k < objs.size(); ++k){
				if (flags[k] == 0){
					float iouUnion = objs[i].iouMinOf(objs[k]);
					if (iouUnion > iou_threshold)
						flags[k] = 1;
				}
			}
		}
		return out;
	}

	vector<BBox> softnms(vector<BBox>& B, float iou_threshold){

		int method = 1;   //1 linear, 2 gaussian, 0 original
		float Nt = iou_threshold;
		float threshold = 0.2;
		float sigma = 0.5;

		std::sort(B.begin(), B.end(), [](const BBox& a, const BBox& b){
			return a.score > b.score;
		});
		
		vector<float> S(B.size());
		for (int i = 0; i < B.size(); ++i)
			S[i] = B[i].score;

		vector<BBox> D;
		while (!B.empty()){

			int m = 0;
			auto M = B[m];
			
			D.push_back(M);
			B.erase(B.begin() + m);
			S.erase(S.begin() + m);
			
			for (int i = (int)B.size() - 1; i >= 0; --i){

				float ov = M.iouOf(B[i]);
				float weight = 1;

				if (method == 1){ //linear
					if (ov > Nt)
						weight = 1 - ov;

				}else if (method == 2){ //gaussian
					weight = exp(-(ov * ov) / sigma);
				}
				else {
					//original nms
					if (ov > Nt)
						weight = 0;
				}
				S[i] *= weight;

				if (S[i] < threshold){
					B.erase(B.begin() + i);
					S.erase(S.begin() + i);
				}
			}
		}
		return D;
	}

	bool remove(const string& file){
#ifdef U_OS_WINDOWS
		return DeleteFileA(file.c_str());
#else
		return ::remove(file.c_str()) == 0;
#endif
	}

	bool mkdir(const string& path){
#ifdef U_OS_WINDOWS
		return CreateDirectoryA(path.c_str(), nullptr);
#else
		return ::mkdir(path.c_str(), 0755) == 0;
#endif
	}

	FILE* fopen_mkdirs(const string& path, const string& mode){

		FILE* f = fopen(path.c_str(), mode.c_str());
		if (f) return f;

		int p = path.rfind('/');

#if defined(U_OS_WINDOWS)
		int e = path.rfind('\\');
		p = std::max(p, e);
#endif
		if (p == -1)
			return nullptr;
		
		string directory = path.substr(0, p);
		if (!mkdirs(directory))
			return nullptr;

		return fopen(path.c_str(), mode.c_str());
	}

	bool moveTo(const string& src, const string& dst){
#if defined(U_OS_WINDOWS)
		return ::MoveFileA(src.c_str(), dst.c_str());
#elif defined(U_OS_LINUX)
		return rename(src.c_str(), dst.c_str()) == 0;
#endif
	}

	bool copyTo(const string& src, const string& dst){
#if defined(U_OS_WINDOWS)
		return ::CopyFileA(src.c_str(), dst.c_str(), false);
#elif defined(U_OS_LINUX)
		FILE* i = fopen(src.c_str(), "rb");
		if (!i) return false;

		FILE* o = fopen(dst.c_str(), "wb");
		if (!o){
			fclose(i);
			return false;
		}

		bool ok = true;
		char buffer[1024];
		int rlen = 0;
		while ((rlen = fread(buffer, 1, sizeof(buffer), i)) > 0){
			if (fwrite(buffer, 1, rlen, o) != rlen){
				ok = false;
				break;
			}
		}
		fclose(i);
		fclose(o);
		return ok;
#endif
	}

	bool mkdirs(const string& path){

		if (path.empty()) return false;
		if (exists(path)) return true;

		string _path = path;
		char* dir_ptr = (char*)_path.c_str();
		char* iter_ptr = dir_ptr;
		
		bool keep_going = *iter_ptr != 0;
		while (keep_going){

			if (*iter_ptr == 0)
				keep_going = false;

#ifdef U_OS_WINDOWS
			if (*iter_ptr == '/' || *iter_ptr == '\\' || *iter_ptr == 0){
#else
			if (*iter_ptr == '/' || *iter_ptr == 0){
#endif
				char old = *iter_ptr;
				*iter_ptr = 0;
				if (!exists(dir_ptr)){
					if (!mkdir(dir_ptr))
						return false;
				}
				*iter_ptr = old;
			}
			iter_ptr++;
		}
		return true;
	}

	bool rmtree(const string& directory, bool ignore_fail){

		auto files = findFiles(directory, "*", false);
		auto dirs = findFiles(directory, "*", true);

		bool success = true;
		for (int i = 0; i < files.size(); ++i){
			if (::remove(files[i].c_str()) != 0){
				success = false;

				if (!ignore_fail){
					return false;
				}
			}
		}

		dirs.insert(dirs.begin(), directory);
		for (int i = (int)dirs.size() - 1; i >= 0; --i){

#ifdef U_OS_WINDOWS
			if (!::RemoveDirectoryA(dirs[i].c_str())){
#else
			if (::rmdir(dirs[i].c_str()) != 0){
#endif
				success = false;
				if (!ignore_fail)
					return false;
			}
		}
		return success;
	}

	void setRandomSeed(int seed){
		srand(seed);
		getRandom().reset(new cv::RNG(seed));
	}

	float randrf(float low, float high){
		if (high < low) std::swap(low, high);
		return getRandom()->uniform(low, high);
	}

	cv::Rect randbox(cv::Size size, cv::Size limit){
		int x = randr(0, limit.width - size.width);
		int y = randr(0, limit.height - size.height);
		return cv::Rect(x, y, size.width, size.height);
	}

	int randr(int high){
		int low = 0;
		if (high < low) std::swap(low, high);
		return randr(low, high);
	}

	int randr(int low, int high){
		if (high < low) std::swap(low, high);
		return getRandom()->uniform(low, high);
	}

	int randr_exclude(int mi, int mx, int exclude){
		if (mi > mx) std::swap(mi, mx);

		if (mx == mi)
			return mi;

		int sel = 0;
		do{
			sel = randr(mi, mx);
		} while (sel == exclude);
		return sel;
	}

	static cv::Scalar HSV2RGB(const float h, const float s, const float v) {
		const int h_i = static_cast<int>(h * 6);
		const float f = h * 6 - h_i;
		const float p = v * (1 - s);
		const float q = v * (1 - f*s);
		const float t = v * (1 - (1 - f) * s);
		float r, g, b;
		switch (h_i) {
		case 0:r = v; g = t; b = p;break;
		case 1:r = q; g = v; b = p;break;
		case 2:r = p; g = v; b = t;break;
		case 3:r = p; g = q; b = v;break;
		case 4:r = t; g = p; b = v;break;
		case 5:r = v; g = p; b = q;break;
		default:r = 1; g = 1; b = 1;break;}
		return cv::Scalar(r * 255, g * 255, b * 255);
	}

	vector<cv::Scalar> randColors(int size){
		vector<cv::Scalar> colors;
		cv::RNG rng(5);
		for (int i = 0; i < size; ++i)
			colors.push_back(HSV2RGB(rng.uniform(0.f, 1.f), 1, 1));
		return colors;
	}

	cv::Scalar randColor(int label, int size){
		static mutex lock_;
		static vector<cv::Scalar> colors;

		if (colors.empty()){
			std::unique_lock<mutex> l(lock_);
			if (colors.empty())
				colors = randColors(size);
		}
		return colors[label % colors.size()];
	}

	const vector<string>& vocLabels(){
		static vector<string> voclabels{
			"aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair",
			"cow", "diningtable", "dog", "horse",
			"motorbike", "person", "pottedplant",
			"sheep", "sofa", "train", "tvmonitor"
		};
		return voclabels;
	}

	vector<int> seque(int begin, int end){

		if (end < begin) std::swap(begin, end);

		int num = end - begin;
		vector<int> out(num);
		for (int i = 0; i < num; ++i)
			out[i] = i + begin;

		return out;
	}

	vector<int> seque(int end){
		return seque(0, end);
	}

	vector<int> shuffleSeque(int begin, int end){
		auto out = seque(begin, end);
		shuffle(out);
		return out;
	}

	vector<int> shuffleSeque(int end){
		return shuffleSeque(0, end);
	}

	int vocLabel(const string& name){
		static map<string, int> labelmap;
		static mutex lock_;
		if (labelmap.empty()){

			std::unique_lock<mutex> l(lock_);
			if (labelmap.empty()){
				auto labels = vocLabels();
				for (int i = 0; i < labels.size(); ++i)
					labelmap[labels[i]] = i;
			}
		}

		auto itr = labelmap.find(name);
		if (itr == labelmap.end()){
			printf("**********name[%s] not in labelmap.\n", name.c_str());
			return -1;
		}
		return itr->second;
	}

	cv::Mat loadMatrix(FILE* f){

		if (!f) return cv::Mat();
		cv::Mat matrix;
		int info[4];
		if (fread(info, 1, sizeof(info), f) != sizeof(info))
			return matrix;

		//flag must match
		//CV_Assert(info[0] == 0xCCABABCC);
		if (info[0] != 0xCCABABCC)
			return matrix;

		int dims[32] = { -1 };
		if (fread(dims, 1, info[1] * sizeof(int), f) != info[1] * sizeof(int))
			return matrix;

		matrix.create(info[1], dims, info[2]);
		bool ok = fread(matrix.data, 1, info[3], f) == info[3];
		if (!ok) matrix.release();
		return matrix;
	}

	bool saveMatrix(FILE* f, const cv::Mat& m){

		if (!f) return false;
		int total;
		cv::Mat w = m;

		if (m.isSubmatrix()){
			//如果是子矩阵，则克隆一个
			w = m.clone();
		}
		else if (m.dims == 2){
			//如果是图像那样存在对齐数据的二维矩阵，那么会由于对齐，也需要clone一个
			if (m.step.p[1] * m.size[1] != m.step.p[0])
				w = m.clone();
		}

		total = w.size[0] * w.step.p[0];

		//dim, type
		int info[] = { (int)0xCCABABCC, m.dims, m.type(), total };
		int wCount = fwrite(info, 1, sizeof(info), f);
		if (wCount != sizeof(info))
			return false;

		fwrite(m.size, 1, sizeof(int) * m.dims, f);
		return fwrite(w.data, 1, total, f) == total;
	}

	cv::Mat loadMatrix(const string& file)
	{
		cv::Mat matrix;
		FILE* f = fopen(file.c_str(), "rb");
		if (!f) return matrix;

		matrix = loadMatrix(f);
		fclose(f);
		return matrix;
	}

	bool saveMatrix(const string& file, const cv::Mat& m){

		FILE* f = fopen(file.c_str(), "wb");
		if (!f) return false;

		bool ok = saveMatrix(f, m);
		fclose(f);
		return ok;
	}

#if defined(U_OS_LINUX)
	typedef struct _GUID {
		unsigned int Data1;
		unsigned short Data2;
		unsigned short Data3;
		unsigned char Data4[8];
	} GUID;
#endif

	//返回32位的大写字母的uuid
	string uuid(){

#if defined(HAS_UUID)

		GUID guid;
#if defined(U_OS_WINDOWS)
		CoCreateGuid(&guid);
#else
		uuid_generate(reinterpret_cast<unsigned char *>(&guid));
#endif

		char buf[33] = { 0 };
#if defined(U_OS_LINUX)
		snprintf(
#else // MSVC
		_snprintf_s(
#endif
			buf,
			sizeof(buf),
			"%08X%04X%04X%02X%02X%02X%02X%02X%02X%02X%02X",
			guid.Data1, guid.Data2, guid.Data3,
			guid.Data4[0], guid.Data4[1],
			guid.Data4[2], guid.Data4[3],
			guid.Data4[4], guid.Data4[5],
			guid.Data4[6], guid.Data4[7]);
		return std::string(buf);
#else
		throw "not implement uuid function";
		return "";
#endif
	}

	BinIO::BinIO(const string& file, const string& mode, bool mkparents){
		openFile(file, mode, mkparents);
	}

	BinIO::~BinIO(){
		close();
	}

	bool BinIO::opened(){
		if (flag_ == FileIO)
			return f_ != nullptr;
		else if (flag_ == MemoryRead)
			return memoryRead_ != nullptr;
		else if (flag_ == MemoryWrite)
			return true;
		return false;
	}

	bool BinIO::openFile(const string& file, const string& mode, bool mkparents){

		close();
		if (mode.empty())
			return false;

		bool hasReadMode = false;
		string mode_ = mode;
		bool hasBinary = false;
		for (int i = 0; i < mode_.length(); ++i){
			if (mode_[i] == 'b'){
				hasBinary = true;
				break;
			}

			if (mode_[i] == 'r')
				hasReadMode = true;
		}

		if (!hasBinary){
			if (mode_.length() == 1){
				mode_.push_back('b');
			}
			else if (mode_.length() == 2){
				mode_.insert(mode_.begin() + 1, 'b');
			}
		}

		if (mkparents)
			f_ = fopen_mkdirs(file, mode_);
		else
			f_ = fopen(file.c_str(), mode_.c_str());
		flag_ = FileIO;

		readModeEndSEEK_ = 0;
		if (hasReadMode && f_ != nullptr){
			//获取他的end
			fseek(f_, 0, SEEK_END);
			readModeEndSEEK_ = ftell(f_);
			fseek(f_, 0, SEEK_SET);
		}
		return opened();
	}

	void BinIO::close(){

		if (flag_ == FileIO) {
			readModeEndSEEK_ = 0;
			if (f_) {
				fclose(f_);
				f_ = nullptr;
			}
		}
		else if (flag_ == MemoryRead) {
			memoryRead_ = nullptr;
			memoryCursor_ = 0;
			memoryLength_ = -1;
		}
		else if (flag_ == MemoryWrite) {
			memoryWrite_.clear();
			memoryCursor_ = 0;
			memoryLength_ = -1;
		}
	}

	string BinIO::readData(int numBytes){
		string output;
		output.resize(numBytes);

		int readlen = read((void*)output.data(), output.size());
		output.resize(readlen);
		return output;
	}

	int BinIO::read(void* pdata, size_t length){

		if (flag_ == FileIO) {
			return fread(pdata, 1, length, f_);
		}
		else if (flag_ == MemoryRead) {

			if (memoryLength_ != -1) {
				
				if (memoryLength_ < memoryCursor_ + length) {
					int remain = memoryLength_ - memoryCursor_;
					if (remain > 0) {
						memcpy(pdata, memoryRead_ + memoryCursor_, remain);
						memoryCursor_ += remain;
						return remain;
					}
					else {
						return -1;
					}
				}
			}
			memcpy(pdata, memoryRead_ + memoryCursor_, length);
			memoryCursor_ += length;
			return length;
		}
		else {
			return -1;
		}
	}
	
	bool BinIO::eof(){
		if (!opened()) return true;

		if (flag_ == FileIO){
			return ftell(f_) >= readModeEndSEEK_ || feof(f_);
		}
		else if (flag_ == MemoryRead){
			return this->memoryCursor_ >= this->memoryLength_;
		}
		else if (flag_ == MemoryWrite){
			return false;
		}
		else {
			INFO("Unsupport flag: %d", flag_);
			return true;
		}
	}

	int BinIO::write(const void* pdata, size_t length){

		if (flag_ == FileIO) {
			return fwrite(pdata, 1, length, f_);
		}
		else if (flag_ == MemoryWrite) {
			memoryWrite_.append((char*)pdata, (char*)pdata + length);
			return length;
		}
		else {
			return -1;
		}
	}

	int BinIO::writeData(const string& data){
		return write(data.data(), data.size());
	}

	BinIO& BinIO::operator >> (string& value){
		//read
		int length = 0;
		(*this) >> length;
		value = readData(length);
		return *this;
	}

	int BinIO::readInt(){
		int value = 0;
		(*this) >> value;
		return value;
	}

	float BinIO::readFloat(){
		float value = 0;
		(*this) >> value;
		return value;
	}

	BinIO& BinIO::operator << (const string& value){
		//write
		(*this) << (int)value.size();
		writeData(value);
		return *this;
	}

	BinIO& BinIO::operator << (const char* value){

		int length = strlen(value);
		(*this) << (int)length;
		write(value, length);
		return *this;
	}

	BinIO& BinIO::operator << (const vector<string>& value){
		(*this) << (int)value.size();
		for (int i = 0; i < value.size(); ++i){
			(*this) << value[i];
		}
		return *this;
	}

	BinIO& BinIO::operator >> (vector<string>& value){
		int num;
		(*this) >> num;

		value.resize(num);
		for (int i = 0; i < value.size(); ++i)
			(*this) >> value[i];
		return *this;
	}

	BinIO& BinIO::operator >> (cv::Mat& value){

		value = loadMatrix(f_);
		return *this;
	}

	BinIO& BinIO::operator << (const cv::Mat& value){
		
		bool ok = saveMatrix(f_, value);
		Assert(ok);
		return *this;
	}

	bool BinIO::openMemoryRead(const void* ptr, int memoryLength) {
		close();

		if (!ptr) return false;
		memoryRead_ = (const char*)ptr;
		memoryCursor_ = 0;
		memoryLength_ = memoryLength;
		flag_ = MemoryRead;
		return true;
	}

	void BinIO::openMemoryWrite() {
		close();

		memoryWrite_.clear();
		memoryCursor_ = 0;
		memoryLength_ = -1;
		flag_ = MemoryWrite;
	}


	//////////////////////////////////////////////////////////////////////////
	//为0时，不cache，为-1时，所有都cache
	FileCache::FileCache(int maxCacheSize){
		maxCacheSize_ = maxCacheSize;
	}

	vector<LabBBox> FileCache::loadxml(const string& file, int* width, int* height, const string& filter){
		return loadxmlFromData(this->loadfile(file), width, height, filter);
	}

	string FileCache::loadfile(const string& file){

		if (maxCacheSize_ == 0)
			return ccutil::loadfile(file);

		std::unique_lock<std::mutex> l(lock_);
		string data;
		if (!hitFile(file)){
			data = ccutil::loadfile(file);

			hits_[file] = data;
			cacheNames_.push_back(file);

			if (maxCacheSize_ > 0 && hits_.size() > maxCacheSize_){

				//random erase
				do{
					int n = ccutil::randr(cacheNames_.size());
					hits_.erase(cacheNames_[n]);
					cacheNames_.erase(cacheNames_.begin() + n);
				} while (hits_.size() > maxCacheSize_);
			}
		}
		else{
			data = hits_[file];
		}
		return data;
	}

	cv::Mat FileCache::loadimage(const string& file, int color){

		cv::Mat image;
		auto data = this->loadfile(file);
		if (data.empty())
			return image;

		try{ image = cv::imdecode(cv::Mat(1, data.size(), CV_8U, (char*)data.data()), color); }
		catch (...){}
		return image;
	}

	bool FileCache::hitFile(const string& file){
		return hits_.find(file) != hits_.end();
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/* MD5 context. */
	typedef struct _MD5_CTX
	{
		unsigned long int state[4]; /* state (ABCD) */
		unsigned long int count[2]; /* number of bits, modulo 2^64 (lsb first) */
		unsigned char buffer[64]; /* input buffer */
	} MD5_CTX;

	/* Constants for MD5Transform routine.*/
	static unsigned char PADDING[64] = {
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	};

	/* F, G, H and I are basic MD5 functions.*/
#define F(x, y, z) (((x) & (y)) | ((~x) & (z))) 
#define G(x, y, z) (((x) & (z)) | ((y) & (~z))) 
#define H(x, y, z) ((x) ^ (y) ^ (z)) 
#define I(x, y, z) ((y) ^ ((x) | (~z))) 

	/* ROTATE_LEFT rotates x left n bits.*/
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n)))) 

	/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4.
	Rotation is separate from addition to prevent recomputation.*/
#define FF(a, b, c, d, x, s, ac) { \
	(a) += F ((b), (c), (d)) + (x) + (unsigned long int)(ac);\
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
		} 
#define GG(a, b, c, d, x, s, ac) { \
	(a) += G ((b), (c), (d)) + (x) + (unsigned long int)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
			} 
#define HH(a, b, c, d, x, s, ac) { \
	(a) += H ((b), (c), (d)) + (x) + (unsigned long int)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
			} 
#define II(a, b, c, d, x, s, ac) { \
	(a) += I ((b), (c), (d)) + (x) + (unsigned long int)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
			} 

	static void MD5Transform(unsigned long int[4], const unsigned char[64]);
	static void MD5Init(MD5_CTX *);
	static void MD5Update(MD5_CTX *, const unsigned char *, unsigned int);
	static void MD5Final(unsigned char[16], MD5_CTX *);
	static void Encode(unsigned char *, unsigned long int *, unsigned int);
	static void Decode(unsigned long int *, const unsigned char *, unsigned int);

	/* MD5 initialization. Begins an MD5 operation, writing a new context.*/
	static void MD5Init(MD5_CTX *context){
		context->count[0] = context->count[1] = 0;
		/* Load magic initialization constants.*/
		context->state[0] = 0x67452301;
		context->state[1] = 0xefcdab89;
		context->state[2] = 0x98badcfe;
		context->state[3] = 0x10325476;
	}

	/* MD5 block update operation. Continues an MD5 message-digest
	operation, processing another message block, and updating the
	context.*/
	static void MD5Update(MD5_CTX *context, /* context */const unsigned char *input, /* input block */unsigned int inputLen /* length of input block */){
		unsigned int i, index, partLen;

		/* Compute number of bytes mod 64 */
		index = (unsigned int)((context->count[0] >> 3) & 0x3F);

		/* Update number of bits */
		if ((context->count[0] += ((unsigned long int)inputLen << 3))
			< ((unsigned long int)inputLen << 3))
			context->count[1]++;
		context->count[1] += ((unsigned long int)inputLen >> 29);

		partLen = 64 - index;

		/* Transform as many times as possible.*/
		if (inputLen >= partLen) {
			memcpy((unsigned char*)&context->buffer[index], (unsigned char*)input, partLen);
			MD5Transform(context->state, context->buffer);

			for (i = partLen; i + 63 < inputLen; i += 64)
				MD5Transform(context->state, &input[i]);

			index = 0;
		}
		else
			i = 0;

		/* Buffer remaining input */
		memcpy((unsigned char*)&context->buffer[index], (unsigned char*)&input[i], inputLen - i);
	}

	/* MD5 finalization. Ends an MD5 message-digest operation, writing the
	the message digest and zeroizing the context.*/
	static void MD5Final(unsigned char digest[16], /* message digest */ MD5_CTX *context /* context */){
		unsigned char bits[8];
		unsigned int index, padLen;

		/* Save number of bits */
		Encode(bits, context->count, 8);

		/* Pad out to 56 mod 64.
		*/
		index = (unsigned int)((context->count[0] >> 3) & 0x3f);
		padLen = (index < 56) ? (56 - index) : (120 - index);
		MD5Update(context, PADDING, padLen);

		/* Append length (before padding) */
		MD5Update(context, bits, 8);

		/* Store state in digest */
		Encode(digest, context->state, 16);

		/* Zeroize sensitive information.*/
		memset((unsigned char*)context, 0, sizeof(*context));
	}

	/* MD5 basic transformation. Transforms state based on block.*/
	static void MD5Transform(unsigned long int state[4], const unsigned char block[64]){
		unsigned long int a = state[0], b = state[1], c = state[2], d = state[3], x[16];

		Decode(x, block, 64);

		/* Round 1 */
		FF(a, b, c, d, x[0], 7, 0xd76aa478); /* 1 */
		FF(d, a, b, c, x[1], 12, 0xe8c7b756); /* 2 */
		FF(c, d, a, b, x[2], 17, 0x242070db); /* 3 */
		FF(b, c, d, a, x[3], 22, 0xc1bdceee); /* 4 */
		FF(a, b, c, d, x[4], 7, 0xf57c0faf); /* 5 */
		FF(d, a, b, c, x[5], 12, 0x4787c62a); /* 6 */
		FF(c, d, a, b, x[6], 17, 0xa8304613); /* 7 */
		FF(b, c, d, a, x[7], 22, 0xfd469501); /* 8 */
		FF(a, b, c, d, x[8], 7, 0x698098d8); /* 9 */
		FF(d, a, b, c, x[9], 12, 0x8b44f7af); /* 10 */
		FF(c, d, a, b, x[10], 17, 0xffff5bb1); /* 11 */
		FF(b, c, d, a, x[11], 22, 0x895cd7be); /* 12 */
		FF(a, b, c, d, x[12], 7, 0x6b901122); /* 13 */
		FF(d, a, b, c, x[13], 12, 0xfd987193); /* 14 */
		FF(c, d, a, b, x[14], 17, 0xa679438e); /* 15 */
		FF(b, c, d, a, x[15], 22, 0x49b40821); /* 16 */

		/* Round 2 */
		GG(a, b, c, d, x[1], 5, 0xf61e2562); /* 17 */
		GG(d, a, b, c, x[6], 9, 0xc040b340); /* 18 */
		GG(c, d, a, b, x[11], 14, 0x265e5a51); /* 19 */
		GG(b, c, d, a, x[0], 20, 0xe9b6c7aa); /* 20 */
		GG(a, b, c, d, x[5], 5, 0xd62f105d); /* 21 */
		GG(d, a, b, c, x[10], 9, 0x2441453); /* 22 */
		GG(c, d, a, b, x[15], 14, 0xd8a1e681); /* 23 */
		GG(b, c, d, a, x[4], 20, 0xe7d3fbc8); /* 24 */
		GG(a, b, c, d, x[9], 5, 0x21e1cde6); /* 25 */
		GG(d, a, b, c, x[14], 9, 0xc33707d6); /* 26 */
		GG(c, d, a, b, x[3], 14, 0xf4d50d87); /* 27 */
		GG(b, c, d, a, x[8], 20, 0x455a14ed); /* 28 */
		GG(a, b, c, d, x[13], 5, 0xa9e3e905); /* 29 */
		GG(d, a, b, c, x[2], 9, 0xfcefa3f8); /* 30 */
		GG(c, d, a, b, x[7], 14, 0x676f02d9); /* 31 */
		GG(b, c, d, a, x[12], 20, 0x8d2a4c8a); /* 32 */

		/* Round 3 */
		HH(a, b, c, d, x[5], 4, 0xfffa3942); /* 33 */
		HH(d, a, b, c, x[8], 11, 0x8771f681); /* 34 */
		HH(c, d, a, b, x[11], 16, 0x6d9d6122); /* 35 */
		HH(b, c, d, a, x[14], 23, 0xfde5380c); /* 36 */
		HH(a, b, c, d, x[1], 4, 0xa4beea44); /* 37 */
		HH(d, a, b, c, x[4], 11, 0x4bdecfa9); /* 38 */
		HH(c, d, a, b, x[7], 16, 0xf6bb4b60); /* 39 */
		HH(b, c, d, a, x[10], 23, 0xbebfbc70); /* 40 */
		HH(a, b, c, d, x[13], 4, 0x289b7ec6); /* 41 */
		HH(d, a, b, c, x[0], 11, 0xeaa127fa); /* 42 */
		HH(c, d, a, b, x[3], 16, 0xd4ef3085); /* 43 */
		HH(b, c, d, a, x[6], 23, 0x4881d05); /* 44 */
		HH(a, b, c, d, x[9], 4, 0xd9d4d039); /* 45 */
		HH(d, a, b, c, x[12], 11, 0xe6db99e5); /* 46 */
		HH(c, d, a, b, x[15], 16, 0x1fa27cf8); /* 47 */
		HH(b, c, d, a, x[2], 23, 0xc4ac5665); /* 48 */

		/* Round 4 */
		II(a, b, c, d, x[0], 6, 0xf4292244); /* 49 */
		II(d, a, b, c, x[7], 10, 0x432aff97); /* 50 */
		II(c, d, a, b, x[14], 15, 0xab9423a7); /* 51 */
		II(b, c, d, a, x[5], 21, 0xfc93a039); /* 52 */
		II(a, b, c, d, x[12], 6, 0x655b59c3); /* 53 */
		II(d, a, b, c, x[3], 10, 0x8f0ccc92); /* 54 */
		II(c, d, a, b, x[10], 15, 0xffeff47d); /* 55 */
		II(b, c, d, a, x[1], 21, 0x85845dd1); /* 56 */
		II(a, b, c, d, x[8], 6, 0x6fa87e4f); /* 57 */
		II(d, a, b, c, x[15], 10, 0xfe2ce6e0); /* 58 */
		II(c, d, a, b, x[6], 15, 0xa3014314); /* 59 */
		II(b, c, d, a, x[13], 21, 0x4e0811a1); /* 60 */
		II(a, b, c, d, x[4], 6, 0xf7537e82); /* 61 */
		II(d, a, b, c, x[11], 10, 0xbd3af235); /* 62 */
		II(c, d, a, b, x[2], 15, 0x2ad7d2bb); /* 63 */
		II(b, c, d, a, x[9], 21, 0xeb86d391); /* 64 */

		state[0] += a;
		state[1] += b;
		state[2] += c;
		state[3] += d;

		/* Zeroize sensitive information.*/
		memset((unsigned char*)x, 0, sizeof(x));
	}

	/* Encodes input (unsigned long int) into output (unsigned char). Assumes len is
	a multiple of 4.*/
	static void Encode(unsigned char *output,unsigned long int *input,unsigned int len){
		unsigned int i, j;

		for (i = 0, j = 0; j < len; i++, j += 4) {
			output[j] = (unsigned char)(input[i] & 0xff);
			output[j + 1] = (unsigned char)((input[i] >> 8) & 0xff);
			output[j + 2] = (unsigned char)((input[i] >> 16) & 0xff);
			output[j + 3] = (unsigned char)((input[i] >> 24) & 0xff);
		}
	}

	/* Decodes input (unsigned char) into output (unsigned long int). Assumes len is
	a multiple of 4.*/
	static void Decode(unsigned long int *output, const unsigned char *input, unsigned int len){
		unsigned int i, j;

		for (i = 0, j = 0; j < len; i++, j += 4)
			output[i] = ((unsigned long int)input[j]) | (((unsigned long int)input[j + 1]) << 8) |
			(((unsigned long int)input[j + 2]) << 16) | (((unsigned long int)input[j + 3]) << 24);
	}

	/* Digests a string and prints the result.*/
	static void md5Calc(const void* data, unsigned int len, char* md5_out)
	{
		MD5_CTX context;
		unsigned char digest[16];
		char output1[34];
		int i;

		MD5Init(&context);
		MD5Update(&context, (unsigned char*)data, len);
		MD5Final(digest, &context);

		for (i = 0; i < 16; i++)
		{
			sprintf(&(output1[2 * i]), "%02x", (unsigned char)digest[i]);
			sprintf(&(output1[2 * i + 1]), "%02x", (unsigned char)(digest[i] << 4));
		}

		for (i = 0; i<32; i++)
			md5_out[i] = output1[i];
	}

	string md5(const void* data, int length)
	{
		string out(32, 0);
		md5Calc(data, length, &out[0]);
		return out;
	}

	string md5(const string& data){
		return md5(data.data(), data.size());
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if defined(U_OS_LINUX)
#define __GetTimeBlock						\
	time_t timep;							\
	time(&timep);							\
	tm& t = *(tm*)localtime(&timep);
#endif

#if defined(U_OS_WINDOWS)
#define __GetTimeBlock						\
	tm t;									\
	_getsystime(&t);
#endif

	string nowFormat(const string& fmt) {
		string output;
		__GetTimeBlock;

		char buf[100];
		for (int i = 0; i < fmt.length(); ++i) {
			char c = fmt[i];
			if (c == 'Y') {			//year
				sprintf(buf, "%04d", t.tm_year + 1900);
			}
			else if (c == 'M') {	//Month
				sprintf(buf, "%02d", t.tm_mon + 1);
			}
			else if (c == 'D') {	//Day
				sprintf(buf, "%02d", t.tm_mday);
			}
			else if (c == 'h') {	//hour
				sprintf(buf, "%02d", t.tm_hour);
			}
			else if (c == 'm') {	//minute
				sprintf(buf, "%02d", t.tm_min);
			}
			else if (c == 's') {	//second
				sprintf(buf, "%02d", t.tm_sec);
			}
			else {
				output.push_back(c);
				continue;
			}
			output += buf;
		}
		return output;
	}

	string timeNow(){
		char time_string[20];
		__GetTimeBlock;

		sprintf(time_string, "%04d-%02d-%02d %02d:%02d:%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
		return time_string;
	}

	string dateNow() {
		char time_string[20];
		__GetTimeBlock;

		sprintf(time_string, "%04d-%02d-%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday);
		return time_string;
	}

	static struct Logger{
		mutex logger_lock_;
		string logger_directory;
		LoggerListener logger_listener = nullptr;
		volatile bool has_logger = true;
		FILE* handler = nullptr;
		size_t lines = 0;		//日志长度计数

		void setLoggerSaveDirectory(const string& loggerDirectory) {

			//if logger is stop
			if (!has_logger)
				return;

			std::unique_lock<mutex> l(logger_lock_);
			if (handler != nullptr){
				//对于已经打开的文件，必须关闭，如果要修改目录的话
				fclose(handler);
				handler = nullptr;
			}

			logger_directory = loggerDirectory;

			if (logger_directory.empty())
				logger_directory = ".";

#if defined(U_OS_LINUX)
			if (logger_directory.back() != '/') {
				logger_directory.push_back('/');
			}
#endif

#if defined(U_OS_WINDOWS)
			if (logger_directory.back() != '/' && logger_directory.back() != '\\') {
				logger_directory.push_back('/');
			}
#endif
		}

		virtual ~Logger(){
			if (handler){
				fclose(handler);
				handler = nullptr;
			}
		}
	}__g_logger;

	static bool loggerCatchListener(const char* file, int line, int level, const char* message){
		__log_func(file, line, level, "%s", message);
		return false;
	}

	LoggerListener getCatchLoggerListener(){
		return loggerCatchListener;
	}

	bool hasLogger(){
		return __g_logger.has_logger;
	}

	void setLogger(bool hasLogger){
		__g_logger.has_logger = hasLogger;
	}

	void setLoggerListener(LoggerListener func){
		__g_logger.logger_listener = func;
	}

	void setLoggerSaveDirectory(const string& loggerDirectory) {
		__g_logger.setLoggerSaveDirectory(loggerDirectory);
	}

	void __log_func(const char* file, int line, int level, const char* fmt, ...) {

		if (__g_logger.logger_listener != nullptr){
			//如果返回false，则直接返回，返回true才会写入日志系统
			char buffer[10000];
			va_list vl;
			va_start(vl, fmt);
			vsprintf(buffer, fmt, vl);
			if (!__g_logger.logger_listener(file, line, level, buffer))
				return;
		}

		//if logger is stop
		if (!__g_logger.has_logger)
			return;

		std::unique_lock<mutex> l(__g_logger.logger_lock_);
		string now = timeNow();

		va_list vl;
		va_start(vl, fmt);
		
		char buffer[10000];
		int n = sprintf(buffer, "%s[%s:%s:%d]:", level_string(level), now.c_str(), fileName(file, true).c_str(), line);
		vsprintf(buffer + n, fmt, vl);
		printf("%s\n", buffer);

		if (!__g_logger.logger_directory.empty()) {
			string file = dateNow();
			string savepath = __g_logger.logger_directory + file + ".log";

			if (__g_logger.handler != nullptr){
				if (__g_logger.lines % 100 == 0){
					if (!ccutil::exists(savepath)){
						fclose(__g_logger.handler);
						__g_logger.handler = nullptr;
					}
				}
			}

			if (__g_logger.handler == nullptr){
				__g_logger.handler = fopen_mkdirs(savepath.c_str(), "a+");
			}

			if (__g_logger.handler) {
				fprintf(__g_logger.handler, "%s\n", buffer);
				__g_logger.lines++;
				
				if (__g_logger.lines % 100 == 0){
					//每10行写入到硬盘
					fflush(__g_logger.handler);
				}

				if (level == LFATAL){
					//如果是错误，那么接下来就会结束程序，结束前需要先把文件关闭掉
					fclose(__g_logger.handler);
					__g_logger.handler = nullptr;
				}
			}
			else {
				printf("ERROR: can not open logger file: %s\n", savepath.c_str());
			}
		}
	}

	class ThreadContext {
	public:
		struct Context {
			void* data_ = nullptr;
		};

		Context* getContext(thread::id idd) {
			Context* output = nullptr;
			std::unique_lock<mutex> l(lock_);
			auto iter = contextMapper_.find(idd);
			if (iter != contextMapper_.end()) {
				output = iter->second.get();
			}
			return output;
		}

		Context* getAndCreateContext(thread::id idd) {
			Context* output = nullptr;
			std::unique_lock<mutex> l(lock_);
			auto iter = contextMapper_.find(idd);
			if (iter != contextMapper_.end()) {
				output = iter->second.get();
			}
			else {
				output = new Context();
				contextMapper_[idd].reset(output);
			}
			return output;
		}

	private:
		mutex lock_;
		map<thread::id, shared_ptr<Context>> contextMapper_;
	};

	static shared_ptr<ThreadContext> g_threadContext(new ThreadContext());
	void setThreadContext(void* ptr) {
		g_threadContext->getAndCreateContext(this_thread::get_id())->data_ = ptr;
	}

	void* getThreadContext() {
		auto context = g_threadContext->getContext(this_thread::get_id());
		void* data = nullptr;
		if (!context) 
			data = context->data_;
		return data;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////
};
