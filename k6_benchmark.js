import { SharedArray } from 'k6/data';
import { scenario } from 'k6/execution';
import http from 'k6/http';
import { check } from 'k6';
import { Trend, Counter } from 'k6/metrics';

let genTokenLatency = new Trend('gen_token_latency');
let promptTokenLatency = new Trend('prompt_token_latency');
let genTokenCounter = new Counter('gen_token');
let promptTokenCounter = new Counter('prompt_token');


const url = `http://${__ENV.HOST}/v1/chat/completions`;


const conversations = new SharedArray('users', function () {
  return JSON.parse(open(`${__ENV.DATA}`));
});


export const options = {
  scenarios: {
    login: {
      executor: 'constant-arrival-rate',
      duration: '10s',
      rate: __ENV.RPS || 10,
      preAllocatedVUs: __ENV.N || 100,
    },
  },
};


export default function () {
  const conv = conversations[scenario.iterationInTest % conversations.length].conversations;
  if (conv.length === 0) {
    console.log(`Empty conversation ${scenario.iterationInTest % conversations.length}`);
    return;
  }
  let res = http.post(url, JSON.stringify(
    {
      model: __ENV.MODEL,
      messages: conv.slice(0, conv.length - 1).map((x) => {
        return {
          role: {
            human: "user",
            gpt: "assistant"
          }[x.from],
          content: x.value
        };
      }),
      //max_tokens: conv[conv.length-1].value.length,
      max_tokens: 128,
      top_k: 200,
    }
  ), {
    headers: { 'Content-Type': 'application/json' },
    timeout: "600s",
  });


  check(res, {
    'is status 200': (r) => r.status === 200, 
  });
  let duration = res.timings.duration / 1000;
  if(res.status === 200) {
    let usage = res.json().usage;
    promptTokenLatency.add(duration / usage.prompt_tokens);
    genTokenLatency.add(duration / usage.completion_tokens);
    promptTokenCounter.add(usage.prompt_tokens);
    genTokenCounter.add(usage.completion_tokens);
  }
  else {
    console.log(`Error ${res.status} ${res.body}`);
  }
}
