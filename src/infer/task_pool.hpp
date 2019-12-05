

#ifndef TASK_POOL_HPP
#define TASK_POOL_HPP

#include <memory>
#include <functional>
#include <mutex>
#include <vector>
#include <omp.h>

//Task Pool是一个发挥GPU性能，让前向密集的一个类
template<typename _InOutFormat, typename _Net>
class Taskpool{

public:
	typedef _InOutFormat InOutFormat;

	struct ProcessBatchItem {
		InOutFormat* data;
		volatile bool* processFlag = nullptr;
		ProcessBatchItem(InOutFormat* data, volatile bool* processFlag) {
			this->data = data;
			this->processFlag = processFlag;
		}
	};
	
	typedef std::vector<ProcessBatchItem> ProcessBatch;
	typedef std::vector<InOutFormat*> Batch;
	typedef std::function<void(const std::vector<int>& inputDims, InOutFormat& input)> PreprocessFunc;
	typedef std::function<void(_Net* net, Batch& batch)> ForwardBatchFunc;
	typedef std::function<void(InOutFormat& output)> BackProcessFunc;

	Taskpool(const std::shared_ptr<_Net>& net, 
		const PreprocessFunc& preprocess, 
		const ForwardBatchFunc& forwardBatch, 
		const BackProcessFunc& backProcess, 
		const std::vector<int>& inputDims,
		int max_batch = 4)
		:net_(net), max_batch_(max_batch), preprocess_(preprocess), 
		forwardBatch_(forwardBatch), backProcess_(backProcess), inputDims_(inputDims){
	}

	//多线程调用
	void forward(InOutFormat& inout){

		volatile bool processFlag = false;
		preprocess_(inputDims_, inout);

		//add job
		do{
			std::unique_lock<std::mutex> l(jobs_lock_);
			jobs_.push_back(ProcessBatchItem(&inout, &processFlag));
		}while (0);

		do{
			//run and process job
			std::unique_lock<std::mutex> l(run_lock_);
			{
				if (processFlag) break;

				std::unique_lock<std::mutex> l(jobs_lock_);
				if (jobs_.size() <= max_batch_){
					std::swap(jobs_, pool_);
				}
				else{
					pool_.insert(pool_.begin(), jobs_.begin(), jobs_.begin() + max_batch_);
					jobs_.erase(jobs_.begin(), jobs_.begin() + max_batch_);
				}
			};

			Batch batch(pool_.size());
			for (int i = 0; i < pool_.size(); ++i) {
				batch[i] = pool_[i].data;
				*pool_[i].processFlag = true;
			}

			forwardBatch_(net_.get(), batch);
			pool_.clear();
		} while (0);
		backProcess_(inout);
	}

private:
	std::vector<int> inputDims_;
	std::shared_ptr<_Net> net_;
	ProcessBatch jobs_, pool_;
	std::mutex jobs_lock_, run_lock_;
	int max_batch_ = 0;
	PreprocessFunc preprocess_;
	ForwardBatchFunc forwardBatch_;
	BackProcessFunc backProcess_;
};
#endif //TASK_POOL_HPP